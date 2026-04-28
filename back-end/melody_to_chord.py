"""Melody-to-chord harmonisation.

Estimates chord progressions from a melody MIDI by analyzing note sequences
against diatonic and chromatic templates. Falls back to a simple progression
if melody analysis is insufficient.
"""
from __future__ import annotations

import logging
import numpy as np
from typing import NamedTuple


class HarmonisedChord(NamedTuple):
    """A chord in a harmonic progression."""
    bar: int  # Bar number (0-indexed)
    beat: float  # Beat within bar (0.0 = downbeat)
    root: str  # 'C', 'C#', 'D', etc.
    quality: str  # 'maj', 'min', 'dom7', 'maj7', 'sus4'
    confidence: float  # 0.0 to 1.0


_TONIC_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Diatonic chord roots for major/minor keys (as offsets from tonic)
_MAJOR_DIATONIC_ROOTS = [0, 2, 4, 5, 7, 9, 11]  # C, D, E, F, G, A, B
_MAJOR_DIATONIC_NAMES = ["I", "ii", "iii", "IV", "V", "vi", "vii°"]
_MAJOR_DIATONIC_QUALITIES = ["maj", "min", "min", "maj", "maj", "min", "dim"]

_MINOR_DIATONIC_ROOTS = [0, 2, 3, 5, 7, 8, 10]  # C, D, Eb, F, G, Ab, Bb
_MINOR_DIATONIC_NAMES = ["i", "ii°", "III", "iv", "v", "VI", "VII"]
_MINOR_DIATONIC_QUALITIES = ["min", "dim", "maj", "min", "min", "maj", "maj"]


def _smooth_melody_notes(notes):
    """
    Smooth melody notes: merge consecutive same-pitch notes (melisma), drop artifacts.

    Handles Basic Pitch output where Cantonese vocal melisma creates many short notes.
    Merges notes with identical pitch, drops merged notes < 30ms to remove transcription artifacts.

    Args:
        notes: List of pretty_midi.Note objects with start, end, pitch attributes.

    Returns:
        List of smoothed note objects with same interface.
    """
    if not notes:
        return notes

    smoothed = []
    current_pitch = None
    current_start = None
    current_duration = 0

    for note in notes:
        if note.pitch == current_pitch:
            current_duration += (note.end - note.start)
        else:
            if current_pitch is not None and current_duration >= 0.03:
                smoothed.append(type('Note', (), {
                    'pitch': current_pitch,
                    'start': current_start,
                    'end': current_start + current_duration,
                    'velocity': notes[len(smoothed)].velocity if smoothed else 64
                })())
            current_pitch = note.pitch
            current_start = note.start
            current_duration = note.end - note.start

    if current_pitch is not None and current_duration >= 0.03:
        smoothed.append(type('Note', (), {
            'pitch': current_pitch,
            'start': current_start,
            'end': current_start + current_duration,
            'velocity': 64
        })())

    return smoothed if smoothed else notes


def _get_seventh_quality(
    root_pc: int,
    quality_base: str,
    pitch_class_weight: np.ndarray,
) -> str:
    """
    Detect if a 7th chord should be used based on pitch class histogram.

    Checks if major 7th (11 semitones) or minor 7th (10 semitones) is prominent
    above the chord root, suggesting a 7th extension.

    Args:
        root_pc: Root pitch class (0-11).
        quality_base: Base chord quality ('maj', 'min').
        pitch_class_weight: Normalized pitch class histogram.

    Returns:
        Modified quality with 7th extension if present, else base quality.
    """
    if quality_base not in ("maj", "min"):
        return quality_base

    major_seventh_pc = (root_pc + 11) % 12
    minor_seventh_pc = (root_pc + 10) % 12

    major_seventh_weight = pitch_class_weight[major_seventh_pc]
    minor_seventh_weight = pitch_class_weight[minor_seventh_pc]

    # Score both variants; prefer based on base quality
    if quality_base == "maj":
        # Major chord: prefer maj7 over dom7
        if major_seventh_weight > 0.05:  # Threshold for presence
            return "maj7"
    elif quality_base == "min":
        # Minor chord: prefer min7 over other variants
        if minor_seventh_weight > 0.05:
            return "min7"

    return quality_base


def estimate_chords_from_melody(
    midi_path: str,
    tonic: str = "C",
    mode: str = "maj",
    bars_per_chord: int = 2,
) -> list[HarmonisedChord]:
    """
    Estimate chord progression from melody MIDI.

    Analyzes the melody in 2-bar segments (by default) and chooses the diatonic
    chord whose pitch classes best match the melody notes in that segment.
    Detects 7th chord extensions (maj7, min7, dom7) for emotional richness.
    Uses Viterbi smoothing with ballad-specific transition preferences.

    Args:
        midi_path: Path to melody MIDI file.
        tonic: Key root ('C', 'C#', etc.).
        mode: 'maj' or 'min'.
        bars_per_chord: Bars per chord change (default 2 for typical pop).

    Returns:
        List of HarmonisedChord.
    """
    try:
        from pretty_midi import PrettyMIDI
    except ImportError:
        logging.error("pretty_midi not installed; cannot analyze melody MIDI")
        return _default_progression(tonic, mode)
    
    try:
        pm = PrettyMIDI(midi_path)
        if not pm.instruments or not pm.instruments[0].notes:
            logging.warning("Melody MIDI has no notes; using default progression")
            return _default_progression(tonic, mode)
        
        notes = sorted(pm.instruments[0].notes, key=lambda n: n.start)
        notes = _smooth_melody_notes(notes)
        melody_duration = notes[-1].end
        
        # Estimate tempo from MIDI
        try:
            tempo = float(pm.estimate_tempo())
            if not (40 < tempo < 240):
                tempo = 120.0
        except (TypeError, ValueError):
            tempo = 120.0
        
        # Bar length in seconds
        bar_length = (4.0 * 60.0) / tempo  # 4 beats per bar
        
        # Divide melody into segments
        num_segments = max(1, int(melody_duration / (bar_length * bars_per_chord)))
        
        # Choose diatonic chords for this key/mode
        if mode == "maj":
            diatonic_roots = _MAJOR_DIATONIC_ROOTS
            diatonic_names = _MAJOR_DIATONIC_NAMES
            diatonic_qualities = _MAJOR_DIATONIC_QUALITIES
        else:
            diatonic_roots = _MINOR_DIATONIC_ROOTS
            diatonic_names = _MINOR_DIATONIC_NAMES
            diatonic_qualities = _MINOR_DIATONIC_QUALITIES
        
        # Convert to actual pitch classes
        tonic_pc = _TONIC_NAMES.index(tonic) if tonic in _TONIC_NAMES else 0
        diatonic_pcs = [(tonic_pc + offset) % 12 for offset in diatonic_roots]
        
        # Analyze each segment
        segment_chords = []  # List of (segment_idx, root_idx, quality, score)
        
        for seg_idx in range(num_segments):
            seg_start = seg_idx * bar_length * bars_per_chord
            seg_end = (seg_idx + 1) * bar_length * bars_per_chord
            
            # Collect pitch classes in this segment
            seg_notes = [n for n in notes if seg_start <= n.start < seg_end]
            if not seg_notes:
                segment_chords.append((seg_idx, 0, "maj", 0.3))  # Default to I/i
                continue
            
            # Pitch class histogram (weighted by note duration)
            pitch_class_weight = np.zeros(12, dtype=np.float32)
            for n in seg_notes:
                pc = n.pitch % 12
                if n.start is not None and n.end is not None:
                    weight = max(n.end - n.start, 0.05)  # Duration-weighted
                else:
                    weight = 0.05  # Default weight if start/end are None
                pitch_class_weight[pc] += weight
            
            # Normalize
            if pitch_class_weight.sum() > 1e-6:
                pitch_class_weight /= pitch_class_weight.sum()
            
            # Score each diatonic chord
            best_root_idx = 0
            best_score = 0.0
            best_quality = "maj"

            for root_offset_idx, pc_offset in enumerate(diatonic_roots):
                chord_root_pc = (tonic_pc + pc_offset) % 12
                quality_base = diatonic_qualities[root_offset_idx]

                # Chord notes (root, third, fifth)
                if mode == "maj":
                    chord_interval_offsets = [0, 4, 7] if quality_base in ("maj",) else ([0, 3, 7] if quality_base in ("min",) else [0, 3, 6])
                else:
                    chord_interval_offsets = [0, 3, 7] if quality_base in ("min",) else ([0, 4, 7] if quality_base in ("maj",) else [0, 3, 6])

                chord_pcs = [(chord_root_pc + off) % 12 for off in chord_interval_offsets]

                # Score: sum of pitch class weights for chord tones
                score = sum(pitch_class_weight[pc] for pc in chord_pcs)

                # Detect 7th chord extension
                quality = _get_seventh_quality(chord_root_pc, quality_base, pitch_class_weight)

                if score > best_score:
                    best_score = score
                    best_root_idx = root_offset_idx
                    best_quality = quality

            segment_chords.append((seg_idx, best_root_idx, best_quality, best_score))
        
        # Apply Viterbi smoothing to prefer chord changes at natural points
        smoothed_chords = _smooth_chord_progression(segment_chords, len(diatonic_roots))
        
        # Convert to HarmonisedChord format
        result = []
        for seg_idx, root_idx, quality, conf in smoothed_chords:
            bar = seg_idx * bars_per_chord
            root = _TONIC_NAMES[(tonic_pc + diatonic_roots[root_idx]) % 12]
            result.append(HarmonisedChord(
                bar=bar,
                beat=0.0,
                root=root,
                quality=quality,
                confidence=min(1.0, conf)
            ))
        
        logging.info("Harmonised melody: %d chords estimated from %d bars", len(result), num_segments * bars_per_chord)
        return result
    
    except Exception as e:
        logging.error("Melody harmonisation failed: %s", e)
        return _default_progression(tonic, mode)


def _build_ballad_transition_costs(n_root_options: int) -> np.ndarray:
    """
    Build transition cost matrix favoring ballad-specific progressions.

    In major keys (I=0, ii=1, iii=2, IV=3, V=4, vi=5, vii°=6):
    - Strong ballad cadences (low cost): vi→IV, IV→V, V→I, ii→V
    - Self-transitions: cost 0.1
    - Default: cost 0.4

    In minor keys, similar patterns apply with i, iv, v, VI.

    Args:
        n_root_options: Number of diatonic chords (typically 7).

    Returns:
        Transition cost matrix (n_root_options x n_root_options).
    """
    transition_cost = np.ones((n_root_options, n_root_options), dtype=np.float32) * 0.4

    # Self-transitions: always cheap (stay on same chord)
    for i in range(n_root_options):
        transition_cost[i, i] = 0.1

    # Ballad-specific progressions (assuming major/minor diatonic layout):
    # For any key mode, the indices are consistent:
    # Major: I(0), ii(1), iii(2), IV(3), V(4), vi(5), vii°(6)
    # Minor: i(0), ii°(1), III(2), iv(3), v(4), VI(5), VII(6)

    if n_root_options >= 7:
        # vi→IV: classic ballad opening
        transition_cost[5, 3] = 0.15
        # IV→V: common progression
        transition_cost[3, 4] = 0.20
        # V→I: strong resolution
        transition_cost[4, 0] = 0.15
        # ii→V: common jazz/pop progression
        transition_cost[1, 4] = 0.20
        # IV→I: plagal cadence (also common in ballads)
        transition_cost[3, 0] = 0.25

        # Additional smooth progressions (falling fifths often sound good):
        # I→IV (up a fourth is falling fifth in reverse)
        transition_cost[0, 3] = 0.25
        # V→vi: deceptive cadence
        transition_cost[4, 5] = 0.30
        # iii→vi: can work in some contexts
        transition_cost[2, 5] = 0.35

    return transition_cost


def _smooth_chord_progression(
    segment_scores: list[tuple[int, int, str, float]],
    n_root_options: int,
) -> list[tuple[int, int, str, float]]:
    """
    Apply Viterbi smoothing to chord progression scores.
    Uses ballad-specific transition costs to prefer natural progressions.
    """
    if not segment_scores:
        return []
    if len(segment_scores) == 1:
        return segment_scores

    n_segments = len(segment_scores)

    # Extract initial observations
    observations = np.zeros((n_segments, n_root_options), dtype=np.float32)
    for seg_idx, root_idx, _, score in segment_scores:
        observations[seg_idx, root_idx] = score

    # Build ballad-aware transition cost matrix
    transition_cost = _build_ballad_transition_costs(n_root_options)
    
    # Viterbi forward pass
    log_prob = np.zeros((n_segments, n_root_options), dtype=np.float32)
    path_ptr = np.zeros((n_segments, n_root_options), dtype=np.int32)
    
    log_prob[0, :] = np.log(observations[0, :] + 1e-10)
    
    for seg in range(1, n_segments):
        for root in range(n_root_options):
            # Score from each previous root
            prev_scores = log_prob[seg - 1, :] - np.log(transition_cost[:, root] + 1e-10)
            best_prev = np.argmax(prev_scores)
            path_ptr[seg, root] = best_prev
            log_prob[seg, root] = prev_scores[best_prev] + np.log(observations[seg, root] + 1e-10)
    
    # Backtrack
    path = np.zeros(n_segments, dtype=np.int32)
    path[-1] = np.argmax(log_prob[-1, :])
    for seg in range(n_segments - 2, -1, -1):
        path[seg] = path_ptr[seg + 1, path[seg + 1]]
    
    # Reconstruct result with smoothed roots
    result = []
    for seg_idx, original_root_idx, quality, _ in segment_scores:
        smoothed_root_idx = int(path[seg_idx])
        confidence = observations[seg_idx, smoothed_root_idx]
        result.append((seg_idx, smoothed_root_idx, quality, confidence))
    
    return result


def _default_progression(tonic: str, mode: str) -> list[HarmonisedChord]:
    """Return a simple default progression: I-V-vi-IV (major) or i-v-VI-iv (minor)."""
    if mode == "maj":
        roots = [0, 7, 9, 5]  # C, G, A, F
        qualities = ["maj", "maj", "min", "maj"]
        names = ["I", "V", "vi", "IV"]
    else:
        roots = [0, 7, 8, 5]  # C, G, Ab, F
        qualities = ["min", "min", "maj", "min"]
        names = ["i", "v", "VI", "iv"]
    
    tonic_pc = _TONIC_NAMES.index(tonic) if tonic in _TONIC_NAMES else 0
    
    result = []
    for bar_idx, root_offset, quality in zip(range(0, 8, 2), roots, qualities):
        root_pc = (tonic_pc + root_offset) % 12
        root_name = _TONIC_NAMES[root_pc]
        result.append(HarmonisedChord(
            bar=bar_idx,
            beat=0.0,
            root=root_name,
            quality=quality,
            confidence=0.5
        ))
    
    return result
