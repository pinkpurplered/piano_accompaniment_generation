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


def _calculate_phase_offset(
    notes: list,
    beat_times_sec: list[float],
    tempo: float,
) -> float:
    """
    Calculate how far the melody start is offset from the beat grid.

    If melody starts at a time that's between beat grid points,
    we need to know that offset to align chords properly.

    Args:
        notes: Smoothed melody notes.
        beat_times_sec: Beat grid times.
        tempo: Tempo in BPM.

    Returns:
        Phase offset in seconds to add to segment times.
    """
    if not notes or not beat_times_sec or len(beat_times_sec) < 2:
        return 0.0

    # Find first melody note time
    first_note_time = notes[0].start if notes else 0.0

    # Find which beat grid interval contains this time
    beat_length_sec = 60.0 / tempo
    beat_times = sorted(beat_times_sec)

    # Find the two beat grid points surrounding the first note
    beat_before = None
    beat_after = None

    for i, beat_time in enumerate(beat_times):
        if beat_time <= first_note_time:
            beat_before = beat_time
        if beat_time > first_note_time and beat_after is None:
            beat_after = beat_time
            break

    # If first note is before first beat or after last beat, return 0
    if beat_before is None or beat_after is None:
        return 0.0

    # Calculate fractional position between beats
    frac = (first_note_time - beat_before) / (beat_after - beat_before)

    # If closer to beat_before, offset toward it; if closer to beat_after, offset toward it
    # Small fractional offsets (< 0.1 or > 0.9) suggest alignment to a beat
    # Middle offsets suggest a phase shift that needs compensation

    if 0.3 < frac < 0.7:  # Middle of beat interval = phase offset
        # Offset back to nearest beat
        if frac < 0.5:
            # Closer to beat_before
            offset = -(beat_before + beat_length_sec - first_note_time)
            logging.info(f"📍 Phase offset detected: melody starts {frac*beat_length_sec:.3f}s after beat")
            logging.info(f"   Offsetting segments by {offset:.3f}s to align with beat grid")
            return offset
        else:
            # Closer to beat_after
            offset = beat_after - first_note_time
            logging.info(f"📍 Phase offset detected: melody starts {(1-frac)*beat_length_sec:.3f}s before beat")
            logging.info(f"   Offsetting segments by {offset:.3f}s to align with beat grid")
            return offset

    return 0.0


def _get_bar_from_beat_grid(
    segment_time_sec: float,
    beat_times_sec: list[float],
    beat_numbers: list[int],
    tempo: float,
    bars_per_chord: int = 2,
    first_beat_in_bar: int = 1,
) -> int:
    """
    Map a segment time (in melody MIDI) to a bar number using beat grid.

    For each beat in the grid, calculates which bar it's in, then finds
    the bar that contains the segment time.

    Args:
        segment_time_sec: Time in melody MIDI (seconds from start).
        beat_times_sec: Times of detected beats (already warped to song time).
        beat_numbers: Which beat within bar (1,2,3,4,1,2,...).
        tempo: Tempo in BPM.
        bars_per_chord: Chords change every N bars.
        first_beat_in_bar: What beat the song starts on (1-4).

    Returns:
        Bar number for this segment.
    """
    if not beat_times_sec or not beat_numbers or len(beat_times_sec) < 2:
        # No beat grid; use simple calculation
        bar_length_sec = (4.0 * 60.0) / tempo
        return int(segment_time_sec / (bar_length_sec * bars_per_chord)) * bars_per_chord

    # Calculate beat duration
    beat_length_sec = 60.0 / tempo

    # Find the beat closest to this segment time
    closest_beat_idx = 0
    min_dist = abs(beat_times_sec[0] - segment_time_sec)
    for i, beat_time in enumerate(beat_times_sec):
        dist = abs(beat_time - segment_time_sec)
        if dist < min_dist:
            min_dist = dist
            closest_beat_idx = i

    # Use beat grid to determine bar number
    # Count downbeats (beat_numbers == 1) up to this beat
    downbeat_count = 0
    anacrusis_offset = (first_beat_in_bar - 1) / 4.0  # Fraction of bar before first downbeat

    for i in range(closest_beat_idx + 1):
        if beat_numbers[i] == 1:
            downbeat_count += 1

    # Bar number = number of downbeats passed - 1 (0-indexed)
    bar_num = downbeat_count - 1

    # Round to nearest chord change boundary
    bar_num = (bar_num // bars_per_chord) * bars_per_chord

    logging.debug(f"📍 Segment at {segment_time_sec:.2f}s → beat {closest_beat_idx} "
                  f"(beat {beat_numbers[closest_beat_idx]}) → bar {bar_num}")

    return bar_num


def _calculate_bar_offset(
    beat_times_sec: list[float],
    beat_numbers: list[int],
    tempo: float,
    pickup_shift: int = 0,
) -> int:
    """
    Calculate bar offset to align chords with actual song structure.

    Accounts for anacrusis (pickup) where songs start on beat 2, 3, or 4 instead of beat 1.
    Note: beat_times_sec is pre-warped by youtube_melody.py to start at time 0,
    so we use beat_numbers to detect anacrusis.

    Args:
        beat_times_sec: Times of beats in audio (seconds, pre-warped to start at 0).
        beat_numbers: Beat position within bar for each beat (1,2,3,4,1,2,...).
        tempo: Tempo in BPM (for bar length calculation).
        pickup_shift: Anacrusis offset in sixteenth notes (legacy parameter).

    Returns:
        Bar offset to add to chord bar positions.
    """
    if not beat_times_sec or not beat_numbers:
        # No beat grid; assume song starts at bar 0, beat 1
        return 0

    first_beat_in_bar = beat_numbers[0] if beat_numbers else 1

    # Handle anacrusis: if song starts on beat 2, 3, or 4, it means
    # there's a pickup at the end of the previous bar
    if first_beat_in_bar == 1:
        # Song starts on downbeat - no offset needed
        bar_offset = 0
        logging.info(f"🎵 Song starts on downbeat (beat 1) - chords start at bar 0")
    else:
        # Song starts with anacrusis (e.g., beat 3 of the intro bar)
        # The chords should align to this structure
        # We DON'T offset bars, but the chords will naturally align
        # because the beat grid accounts for it
        bar_offset = 0
        logging.info(f"🎵 Song starts with anacrusis (beat {first_beat_in_bar}) - "
                    f"chords will align to pickup structure")

    return bar_offset


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
    tempo: float | None = None,
    beat_times_sec: list[float] | None = None,
    beat_numbers: list[int] | None = None,
    pickup_shift: int = 0,
) -> list[HarmonisedChord]:
    """
    Estimate chord progression from melody MIDI with beat-aware alignment.

    Analyzes the melody in 2-bar segments (by default) and chooses the diatonic
    chord whose pitch classes best match the melody notes in that segment.
    Detects 7th chord extensions (maj7, min7, dom7) for emotional richness.
    Uses Viterbi smoothing with ballad-specific transition preferences.

    Aligns chords to the actual song structure using beat grid information:
    - Finds where the melody actually starts in the beat grid
    - Adjusts chord bar positions to match the song's actual bars/beats
    - Handles anacrusis (pickup) alignment

    Args:
        midi_path: Path to melody MIDI file.
        tonic: Key root ('C', 'C#', etc.).
        mode: 'maj' or 'min'.
        bars_per_chord: Bars per chord change (default 2 for typical pop).
        tempo: Optional tempo in BPM. If not provided, estimates from MIDI.
        beat_times_sec: Beat grid times from instrumental stem analysis.
                       Used to align chords to actual song structure.
        beat_numbers: Beat position within bar for each beat (1, 2, 3, 4, 1, 2, ...).
        pickup_shift: Anacrusis offset in sixteenth notes (for proper downbeat alignment).

    Returns:
        List of HarmonisedChord aligned to actual song structure.
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

        logging.info("🎵 Melody MIDI: %d notes, %.1f seconds duration", len(notes), melody_duration)
        if beat_times_sec:
            logging.info("🎵 Beat grid available: %d beats, beat_numbers=%s",
                        len(beat_times_sec), beat_numbers[:4] if beat_numbers else "unknown")
        
        # Use provided tempo or estimate from MIDI
        if tempo is None:
            try:
                tempo = float(pm.estimate_tempo())
                if not (40 < tempo < 240):
                    tempo = 120.0
            except (TypeError, ValueError):
                tempo = 120.0
            logging.info("📍 Tempo estimated from MIDI: %.1f BPM", tempo)
        else:
            logging.info("📍 Using provided tempo: %.1f BPM", tempo)

        # Bar length in seconds
        bar_length = (4.0 * 60.0) / tempo  # 4 beats per bar

        # Divide melody into segments
        num_segments = max(1, int(melody_duration / (bar_length * bars_per_chord)))

        # Debug: check if melody duration is suspiciously long
        max_reasonable_duration = 600  # 10 minutes
        if melody_duration > max_reasonable_duration:
            logging.warning("⚠️ Melody MIDI duration %.1f sec is very long (typical song: 3-5 min). "
                           "May contain transcription artifacts or silence. Trimming to %.1f sec.",
                           melody_duration, max_reasonable_duration)
            melody_duration = min(melody_duration, max_reasonable_duration)
            num_segments = max(1, int(melody_duration / (bar_length * bars_per_chord)))

        logging.info("📊 Chord estimation: tempo=%.1f BPM, bar_length=%.2f sec, "
                    "melody_duration=%.1f sec → %d segments (%d-bar chunks)",
                    tempo, bar_length, melody_duration, num_segments, bars_per_chord)
        
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

        # Calculate beat grid phase offset (when does first melody note align with beat grid?)
        phase_offset = _calculate_phase_offset(
            notes,
            beat_times_sec or [],
            tempo
        )

        # Map segment times to beat grid for proper alignment
        segment_times = []
        for seg_idx in range(num_segments):
            seg_time = seg_idx * bar_length * bars_per_chord + phase_offset
            segment_times.append(seg_time)

        # Convert to HarmonisedChord format with beat-grid alignment
        result = []
        for seg_idx, root_idx, quality, conf in smoothed_chords:
            # Find which beat this segment aligns with
            seg_time = segment_times[seg_idx]
            bar_num = _get_bar_from_beat_grid(
                seg_time,
                beat_times_sec or [],
                beat_numbers or [],
                tempo,
                bars_per_chord,
                first_beat_in_bar=beat_numbers[0] if beat_numbers else 1
            )

            root = _TONIC_NAMES[(tonic_pc + diatonic_roots[root_idx]) % 12]
            result.append(HarmonisedChord(
                bar=bar_num,
                beat=0.0,
                root=root,
                quality=quality,
                confidence=min(1.0, conf)
            ))

            if seg_idx < 5:  # Log first 5 chords
                logging.info(f"  Chord {seg_idx}: {root} {quality} at bar {bar_num} (seg_time={seg_time:.2f}s)")
        
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
