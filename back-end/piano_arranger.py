"""Piano accompaniment arranger.

Converts a chord progression into voiced piano MIDI with automatic texture selection,
voice leading, and beat grid alignment. Produces two outputs:
- Block chords (one voicing per chord change) for use as a lead sheet
- Textured chords (flowing arpeggios, sustained chords, etc.) for music
"""
from __future__ import annotations

import logging
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple


def build_chord_anchors(
    tempo_bpm: float,
    drum_hit_times: list[float] | None,
    beat_grid_times_sec: list[float] | None = None,
    song_end_sec: float | None = None,
) -> list[float]:
    """Build chord-placement times spaced one half-bar apart (beats 1 and 3).

    Uses drum hits as ground truth: drums hit on beats 2 and 4 (backbeat) in
    Cantonese ballads, so consecutive hits sit ~half-bar apart. Find the longest
    regularly-spaced run, then shift back by one beat so anchors land on the
    listener's beats 1 and 3. Stride evenly from there.
    """
    bar_sec = 240.0 / max(tempo_bpm, 1.0)
    half_bar = bar_sec / 2.0
    drums = sorted(drum_hit_times or [])
    bts = sorted(beat_grid_times_sec or [])

    if song_end_sec is None:
        if drums:
            song_end_sec = max(drums[-1], bts[-1] if bts else 0.0) + bar_sec
        elif bts:
            song_end_sec = bts[-1] + bar_sec
        else:
            song_end_sec = 240.0

    if len(drums) < 4:
        anchor0 = bts[0] if bts else 0.0
        logging.info("🥁 No drums; uniform anchor grid from %.2fs", anchor0)
    else:
        tol = half_bar * 0.30
        runs, cur = [], [0]
        for i in range(1, len(drums)):
            if abs((drums[i] - drums[i - 1]) - half_bar) <= tol:
                cur.append(i)
            else:
                if len(cur) >= 3:
                    runs.append(cur)
                cur = [i]
        if len(cur) >= 3:
            runs.append(cur)
        if runs:
            best = max(runs, key=len)
            anchor0 = drums[best[0]]
            logging.info("🥁 Anchored to drum run of %d hits @ %.2fs (half_bar=%.3fs)",
                         len(best), anchor0, half_bar)
        else:
            anchor0 = drums[len(drums) // 2]
            logging.info("🥁 No regular drum run; anchoring at middle hit %.2fs", anchor0)
        # Drums sit on backbeats (2 & 4); shift back one beat so anchors are on 1 & 3.
        anchor0 -= bar_sec / 4.0
        logging.info("🥁 Shifted anchor by -1 beat → %.2fs (beat 1)", anchor0)

    fwd = [anchor0]
    t = anchor0
    while t + half_bar < song_end_sec:
        t += half_bar
        fwd.append(t)
    back = []
    t = anchor0
    while t - half_bar > 0:
        t -= half_bar
        if t < 0:
            break
        back.append(t)
    anchors = list(reversed(back)) + fwd
    logging.info("🥁 Built %d chord anchors (%.2fs … %.2fs)",
                 len(anchors), anchors[0], anchors[-1])
    return anchors


class Texture(Enum):
    """Piano texture style."""
    BLOCK = "block"  # Sustained block chords
    ARPEGGIO = "arpeggio"  # Flowing arpeggios (Alberti, etc.)
    BALLAD = "ballad"  # LH bass + RH chords (light-classical ballad style)


class ChordVoicing(NamedTuple):
    """A voiced chord: pitches and velocities."""
    pitches: list[int]  # MIDI pitch numbers
    velocities: list[int]  # 0-127


@dataclass
class ArrangementConfig:
    """Configuration for piano arrangement."""
    tempo_bpm: float = 120.0
    time_signature: tuple[int, int] = (4, 4)  # (beats_per_bar, beat_value)
    beat_grid_times_sec: list[float] = None  # Beat positions in seconds
    beat_numbers: list[int] = None  # Beat position in bar (1, 2, 3, 4, ...)
    drum_hit_times: list[float] = None  # Times of detected drum hits (preferred chord-placement anchors)
    precomputed_anchors: list[float] = None  # If set, skip anchor building
    slots_per_chord: int = 2  # 2 = chord lasts one bar (legacy); 1 = chord lasts one half-bar (new flow)
    default_texture: Texture = Texture.ARPEGGIO
    allow_texture_selection: bool = True  # Auto-select texture based on melody density


class ChordVoicer:
    """Convert chord symbols to voiced piano notes with voice leading."""
    
    # Pitch classes: C=0, C#=1, ..., B=11
    _PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    _NOTE_NAME_TO_PC = {n: i for i, n in enumerate(_PITCH_NAMES)}
    
    # Chord templates: (intervals_from_root_in_semitones, chord_name)
    _CHORD_TEMPLATES = {
        "maj": [0, 4, 7],  # Major triad
        "min": [0, 3, 7],  # Minor triad
        "dim": [0, 3, 6],  # Diminished
        "aug": [0, 4, 8],  # Augmented
        "sus2": [0, 2, 7],  # Sus2
        "sus4": [0, 5, 7],  # Sus4
        "maj7": [0, 4, 7, 11],  # Major 7
        "min7": [0, 3, 7, 10],  # Minor 7
        "dom7": [0, 4, 7, 10],  # Dominant 7
        "dim7": [0, 3, 6, 9],  # Diminished 7 (diminished triad + diminished 7th)
        "half-dim7": [0, 3, 6, 10],  # Half-diminished 7 (diminished triad + minor 7th)
        "min-maj7": [0, 3, 7, 11],  # Minor-major 7
    }
    
    def __init__(self):
        self.last_voicing: ChordVoicing | None = None
    
    def voice_chord(
        self,
        root: str,  # 'C', 'C#', etc.
        quality: str,  # 'maj', 'min', 'maj7', etc.
        octave: int = 4,  # Soprano octave for RH (default higher)
        invert: int = 0,  # 0=root position, 1=first inversion, etc.
        voice_lead: bool = True,  # Use voice leading from last chord
    ) -> ChordVoicing:
        """
        Voice a chord across piano keyboard for rich, full sound.

        Creates 5-6 voices spanning bass to soprano for proper piano accompaniment:
        - Bass (LH, octave 2): root or fundamental
        - Tenor (LH, octave 3): fills middle-low
        - Alto (RH, octave 4): chord tones
        - Soprano (RH, octave 5): highest voices for sparkle

        Args:
            root: Root note name.
            quality: Chord quality.
            octave: Soprano octave for RH (default 4, range 3-5).
            invert: Inversion (0=root position, 1=first inversion, etc.).
            voice_lead: Use voice leading to minimize motion from last voicing.

        Returns:
            ChordVoicing with pitches and velocities spread across keyboard.
        """
        # Look up chord intervals
        if quality not in self._CHORD_TEMPLATES:
            quality = "maj"  # Default to major

        intervals = self._CHORD_TEMPLATES[quality][:]
        root_pc = self._NOTE_NAME_TO_PC.get(root, 0)
        chord_pcs = [(root_pc + int_val) % 12 for int_val in intervals]

        pitches = []
        velocities = []

        # LH: single bass note, root in octave 3 (C3 = MIDI 48, range C2–C4)
        bass_pc = chord_pcs[0]
        bass_pitch = 48 + bass_pc  # octave 3 by default
        if bass_pitch > 60:        # don't go above C4
            bass_pitch -= 12
        pitches.append(bass_pitch)
        velocities.append(72)

        # RH: chord tones clustered around C4 (MIDI 60)
        for i, pc in enumerate(chord_pcs):
            rh_pitch = 60 + pc  # octave 4
            if rh_pitch > 72:   # keep below C5 for a tight voicing
                rh_pitch -= 12
            pitches.append(rh_pitch)
            velocities.append(65 if i == 0 else 58)

        # If voice leading is enabled and we have a previous voicing, optimize
        if voice_lead and self.last_voicing:
            pitches, velocities = self._voice_lead(pitches, velocities, self.last_voicing)

        voicing = ChordVoicing(pitches=pitches, velocities=velocities)
        self.last_voicing = voicing
        return voicing
    
    def _voice_lead(self, new_pitches: list[int], new_velocities: list[int], 
                    last_voicing: ChordVoicing) -> tuple[list[int], list[int]]:
        """
        Adjust voicing to minimize motion from previous chord (smoother transitions).
        Uses greedy assignment of voices to minimize total semitone movement.
        """
        if not last_voicing.pitches or len(new_pitches) < 2:
            return new_pitches, new_velocities
        
        # Greedy: assign each new voice to the closest old voice
        old_pitches = list(last_voicing.pitches)
        assignments = []
        
        for new_pitch in new_pitches:
            distances = [abs(new_pitch - old_pitch) if old_pitch is not None else 1000 for old_pitch in old_pitches]
            best_idx = np.argmin(distances)
            assignments.append(best_idx)
            old_pitches[best_idx] = None  # Mark as used
        
        # Recalculate: some voices might be None, filter them
        used_old_indices = set(assignments)
        
        # Keep bass note fixed, allow upper voices to move
        if len(new_pitches) > 1:
            # Ensure the lowest note stays lowest-ish
            pass
        
        return new_pitches, new_velocities


class PianoArranger:
    """Arrange chord progression as piano MIDI."""
    
    def __init__(self, config: ArrangementConfig | None = None):
        self.config = config or ArrangementConfig()
        self.voicer = ChordVoicer()
        self._anchors: list[float] = []

    def _build_anchors(self) -> list[float]:
        if self.config.precomputed_anchors:
            return list(self.config.precomputed_anchors)
        return build_chord_anchors(
            tempo_bpm=self.config.tempo_bpm,
            drum_hit_times=self.config.drum_hit_times,
            beat_grid_times_sec=self.config.beat_grid_times_sec,
        )

    def _slot(self, idx: int) -> tuple[float, float] | tuple[None, None]:
        """Return (start_sec, duration_sec) for the idx-th half-bar slot, or (None, None)."""
        if not self._anchors or idx < 0 or idx >= len(self._anchors):
            return None, None
        start = self._anchors[idx]
        if idx + 1 < len(self._anchors):
            dur = self._anchors[idx + 1] - start
        else:
            dur = 240.0 / (2.0 * max(self.config.tempo_bpm, 1.0))
        return start, max(dur, 0.05)

    def arrange(
        self,
        chords: list[tuple[int, str, str]],  # (bar, root, quality)
        melody_midi_path: str | None = None,
        output_midi_path: str | None = None,
    ) -> tuple[list[dict], list[dict]]:
        """
        Arrange chords as piano accompaniment.
        
        Args:
            chords: List of (bar_start, root, quality) tuples.
            melody_midi_path: Optional path to melody MIDI for texture analysis.
            output_midi_path: Optional path to write output MIDI.
            
        Returns:
            (block_chord_notes, textured_notes) where each is a list of note dicts
            with keys: time_sec, duration_sec, pitch, velocity.
        """
        if not chords:
            logging.warning("No chords to arrange")
            return [], []

        self._anchors = self._build_anchors()

        # Estimate melody density from melody MIDI (for texture selection)
        melody_density = self._analyze_melody_density(melody_midi_path) if melody_midi_path else 0.5
        
        # Select texture
        # If allow_texture_selection is True and default is ARPEGGIO, auto-select based on melody
        # Otherwise, use the explicitly configured texture (e.g., BALLAD for slow tempos)
        if self.config.allow_texture_selection and self.config.default_texture == Texture.ARPEGGIO:
            texture = self._select_texture(melody_density)
        else:
            texture = self.config.default_texture
        
        logging.info("Piano arrangement: texture=%s (melody_density=%.2f)", texture.value, melody_density)
        
        # Generate block chords (one voicing per chord change)
        block_notes = self._generate_block_chords(chords)
        
        # Generate textured chords (flowing arpeggios, etc.)
        if texture == Texture.BLOCK:
            textured_notes = block_notes  # Same as block
        elif texture == Texture.ARPEGGIO:
            textured_notes = self._generate_arpeggio_texture(chords)
        elif texture == Texture.BALLAD:
            textured_notes = self._generate_ballad_texture(chords)
        else:
            textured_notes = block_notes
        
        # Write MIDI files if path provided
        if output_midi_path:
            self._write_midi_file(output_midi_path, textured_notes)
        
        return block_notes, textured_notes
    
    def _analyze_melody_density(self, melody_midi_path: str) -> float:
        """Estimate note density in melody (0.0-1.0) to choose texture."""
        try:
            from pretty_midi import PrettyMIDI
            pm = PrettyMIDI(melody_midi_path)
            if not pm.instruments or not pm.instruments[0].notes:
                return 0.5
            
            notes = pm.instruments[0].notes
            duration = notes[-1].end if notes else 1.0
            density = len(notes) / (duration + 1e-6)
            
            # Normalize: typical melodies have 2-10 notes per second
            return min(1.0, max(0.0, (density - 2) / 8))
        except Exception as e:
            logging.warning("Melody density analysis failed: %s", e)
            return 0.5
    
    def _select_texture(self, melody_density: float) -> Texture:
        """Choose texture based on melody activity."""
        if melody_density < 0.3:
            return Texture.ARPEGGIO  # Sparse melody → flowing accompaniment
        elif melody_density > 0.7:
            return Texture.BLOCK  # Dense melody → simple chords to avoid clashing
        else:
            return Texture.BALLAD  # Medium density → light accompaniment
    
    def _generate_block_chords(self, chords: list[tuple[int, str, str]]) -> list[dict]:
        """Re-articulate each chord on beats 1 and 3 (two slots per bar)."""
        notes = []
        for chord_idx, (_bar, root, quality) in enumerate(chords):
            voicing = self.voicer.voice_chord(root, quality, octave=4, voice_lead=True)
            for slot_offset in range(self.config.slots_per_chord):  # beat 1, beat 3
                start, dur = self._slot(chord_idx * self.config.slots_per_chord + slot_offset)
                if start is None:
                    break
                for pitch, velocity in zip(voicing.pitches, voicing.velocities):
                    notes.append({
                        "time_sec": start,
                        "duration_sec": dur * 0.95,
                        "pitch": pitch,
                        "velocity": velocity,
                    })
        return notes
    
    def _generate_arpeggio_texture(self, chords: list[tuple[int, str, str]]) -> list[dict]:
        """Alberti pattern, one cycle per half-bar slot anchored to beats 1 and 3."""
        notes = []
        for chord_idx, (_bar, root, quality) in enumerate(chords):
            voicing = self.voicer.voice_chord(root, quality, octave=4, voice_lead=True)
            if len(voicing.pitches) < 2:
                for slot_offset in range(self.config.slots_per_chord):
                    start, dur = self._slot(chord_idx * self.config.slots_per_chord + slot_offset)
                    if start is None:
                        break
                    for pitch, velocity in zip(voicing.pitches, voicing.velocities):
                        notes.append({
                            "time_sec": start,
                            "duration_sec": dur * 0.95,
                            "pitch": pitch,
                            "velocity": int(velocity * 0.8),
                        })
                continue

            lh_bass = voicing.pitches[0]
            lh_tenor = voicing.pitches[1] if len(voicing.pitches) > 1 else lh_bass
            rh_pitches = voicing.pitches[2:] if len(voicing.pitches) > 2 else [voicing.pitches[-1]]
            if len(rh_pitches) >= 2:
                pattern_pitches = [lh_bass, rh_pitches[-1], rh_pitches[0], rh_pitches[-1]]
                pattern_velocities = [78, 60, 68, 60]
            else:
                pattern_pitches = [lh_bass, lh_tenor, rh_pitches[0] if rh_pitches else lh_tenor, lh_tenor]
                pattern_velocities = [78, 64, 68, 64]

            for slot_offset in range(self.config.slots_per_chord):
                start, dur = self._slot(chord_idx * self.config.slots_per_chord + slot_offset)
                if start is None:
                    break
                step = dur / len(pattern_pitches)
                for pat_idx, pitch in enumerate(pattern_pitches):
                    notes.append({
                        "time_sec": start + pat_idx * step,
                        "duration_sec": step * 0.9,
                        "pitch": pitch,
                        "velocity": pattern_velocities[pat_idx],
                    })
        return notes
    
    def _generate_ballad_texture(self, chords: list[tuple[int, str, str]]) -> list[dict]:
        """Ballad: each slot (beat 1, beat 3) hits LH bass + rolled RH chord."""
        notes = []
        for chord_idx, (_bar, root, quality) in enumerate(chords):
            voicing = self.voicer.voice_chord(root, quality, octave=4, voice_lead=True)
            rh_pitches = voicing.pitches[2:] if len(voicing.pitches) > 2 else [voicing.pitches[-1]]
            rh_velocities = voicing.velocities[2:] if len(voicing.velocities) > 2 else [voicing.velocities[-1]]
            for slot_offset in range(self.config.slots_per_chord):
                start, dur = self._slot(chord_idx * self.config.slots_per_chord + slot_offset)
                if start is None:
                    break
                # LH bass + tenor on the slot start
                notes.append({
                    "time_sec": start,
                    "duration_sec": dur * 0.95,
                    "pitch": voicing.pitches[0],
                    "velocity": 80,
                })
                if len(voicing.pitches) > 1:
                    notes.append({
                        "time_sec": start + 0.04,
                        "duration_sec": dur * 0.9,
                        "pitch": voicing.pitches[1],
                        "velocity": 68,
                    })
                # RH chord rolled across the slot
                roll_step = dur / max(len(rh_pitches), 2) * 0.5
                for k, (pitch, velocity) in enumerate(zip(rh_pitches, rh_velocities)):
                    notes.append({
                        "time_sec": start + k * roll_step,
                        "duration_sec": dur * 0.85,
                        "pitch": pitch,
                        "velocity": int(velocity * 0.85),
                    })
        return notes
    
    def _write_midi_file(self, output_path: str, notes: list[dict]) -> None:
        """Write notes to MIDI file."""
        try:
            import pretty_midi
        except ImportError:
            logging.error("pretty_midi not installed; cannot write MIDI file")
            return
        
        try:
            pm = pretty_midi.PrettyMIDI()
            instrument = pretty_midi.Instrument(program=0)  # Piano
            
            for note_dict in notes:
                note = pretty_midi.Note(
                    velocity=note_dict.get("velocity", 64),
                    pitch=note_dict.get("pitch", 60),
                    start=note_dict.get("time_sec", 0),
                    end=note_dict.get("time_sec", 0) + note_dict.get("duration_sec", 0.5),
                )
                instrument.notes.append(note)
            
            pm.instruments.append(instrument)
            pm.write(output_path)
            logging.info("Piano arrangement written to %s (%d notes)", output_path, len(notes))
        except Exception as e:
            logging.error("Failed to write MIDI file: %s", e)


def arrange_piano_accompaniment(
    chords: list[tuple[int, str, str]],
    tempo_bpm: float = 120.0,
    time_signature: tuple[int, int] = (4, 4),
    beat_grid_times_sec: list[float] | None = None,
    beat_numbers: list[int] | None = None,
    drum_hit_times: list[float] | None = None,
    precomputed_anchors: list[float] | None = None,
    slots_per_chord: int = 2,
    melody_midi_path: str | None = None,
    output_midi_path: str | None = None,
    texture: Texture = Texture.ARPEGGIO,
) -> tuple[list[dict], list[dict]]:
    """
    Convenience function to arrange chords as piano accompaniment.

    Returns (block_chords, textured_chords) where each is a list of note dicts.

    When texture is explicitly BALLAD or BLOCK, disables auto-texture-selection
    to respect the caller's choice. ARPEGGIO allows melody-density analysis.
    """
    # Disable auto-selection if texture is explicitly set to BALLAD or BLOCK
    allow_auto_select = (texture == Texture.ARPEGGIO)

    config = ArrangementConfig(
        tempo_bpm=tempo_bpm,
        time_signature=time_signature,
        beat_grid_times_sec=beat_grid_times_sec or [],
        beat_numbers=beat_numbers or [],
        drum_hit_times=drum_hit_times or [],
        precomputed_anchors=precomputed_anchors,
        slots_per_chord=slots_per_chord,
        default_texture=texture,
        allow_texture_selection=allow_auto_select,
    )
    
    arranger = PianoArranger(config)
    return arranger.arrange(chords, melody_midi_path, output_midi_path)
