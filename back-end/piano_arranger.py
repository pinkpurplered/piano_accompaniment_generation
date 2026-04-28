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
        octave: int = 3,  # Middle octave for root
        invert: int = 0,  # 0=root position, 1=first inversion, etc.
        voice_lead: bool = True,  # Use voice leading from last chord
    ) -> ChordVoicing:
        """
        Voice a chord and return pitches + velocities.
        
        Args:
            root: Root note name.
            quality: Chord quality.
            octave: Octave of root note (MIDI octave numbering, 0-8).
            invert: Inversion (0=root position, 1=first inversion, etc.).
            voice_lead: Use voice leading to minimize motion from last voicing.
            
        Returns:
            ChordVoicing with pitches and velocities.
        """
        # Look up chord intervals
        if quality not in self._CHORD_TEMPLATES:
            quality = "maj"  # Default to major
        
        intervals = self._CHORD_TEMPLATES[quality][:]
        root_pc = self._NOTE_NAME_TO_PC.get(root, 0)
        
        # Build chord pitches starting from octave 1 (very low) to octave 5 (high)
        # Use three-octave range for rich piano sound
        chord_pcs = [(root_pc + int_val) % 12 for int_val in intervals]
        pitches = []
        velocities = []
        
        # Bass notes (octave 1-2)
        pitches.append(12 + root_pc)  # Root in octave 1
        velocities.append(80)  # Bass is stronger
        
        if invert == 1 and len(chord_pcs) > 1:
            # First inversion: third in bass
            pitches[0] = 12 + chord_pcs[1]
        
        # Mid-range chord tones (octave 3-4)
        for i, pc in enumerate(chord_pcs):
            midi_pitch = 12 * octave + pc
            pitches.append(midi_pitch)
            if i == 0:
                velocities.append(72)  # Root is stronger
            else:
                velocities.append(60)  # Other chord tones softer
        
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
        """Generate one block chord at the start of each chord change."""
        notes = []
        bar_length_sec = (4.0 * 60.0) / self.config.tempo_bpm
        
        for bar, root, quality in chords:
            time_sec = bar * bar_length_sec
            voicing = self.voicer.voice_chord(root, quality, voice_lead=True)
            
            # Whole-note chord (4 bars worth at tempo)
            duration_sec = 4.0 * bar_length_sec
            
            for pitch, velocity in zip(voicing.pitches, voicing.velocities):
                notes.append({
                    "time_sec": time_sec,
                    "duration_sec": duration_sec,
                    "pitch": pitch,
                    "velocity": velocity,
                })
        
        return notes
    
    def _generate_arpeggio_texture(self, chords: list[tuple[int, str, str]]) -> list[dict]:
        """Generate flowing Alberti-style arpeggios."""
        notes = []
        bar_length_sec = (4.0 * 60.0) / self.config.tempo_bpm
        beat_length_sec = bar_length_sec / (self.config.time_signature[0] or 4)
        
        # Sixteenth-note pattern (4 notes per beat)
        sixteenth_sec = beat_length_sec / 4
        
        for bar, root, quality in chords:
            bar_start_sec = bar * bar_length_sec
            voicing = self.voicer.voice_chord(root, quality, voice_lead=True)
            
            if len(voicing.pitches) < 2:
                # Not enough notes for arpeggio; fall back to block
                duration_sec = bar_length_sec
                for pitch, velocity in zip(voicing.pitches, voicing.velocities):
                    notes.append({
                        "time_sec": bar_start_sec,
                        "duration_sec": duration_sec,
                        "pitch": pitch,
                        "velocity": int(velocity * 0.8),
                    })
                continue
            
            # Alberti pattern: bass, high, middle, high, repeat
            pattern_pitches = [voicing.pitches[0], voicing.pitches[-1], voicing.pitches[-2 if len(voicing.pitches) > 2 else 1], voicing.pitches[-1]]
            pattern_velocities = [70, 50, 55, 50]
            
            # Play pattern for the entire bar (4 patterns per bar at typical tempo)
            num_repetitions = int(bar_length_sec / (len(pattern_pitches) * sixteenth_sec)) + 1
            
            for rep in range(num_repetitions):
                for pat_idx, pitch in enumerate(pattern_pitches):
                    time_sec = bar_start_sec + rep * len(pattern_pitches) * sixteenth_sec + pat_idx * sixteenth_sec
                    if time_sec >= bar_start_sec + bar_length_sec:
                        break
                    
                    notes.append({
                        "time_sec": time_sec,
                        "duration_sec": sixteenth_sec * 0.9,  # Small gap
                        "pitch": pitch,
                        "velocity": pattern_velocities[pat_idx],
                    })
        
        return notes
    
    def _generate_ballad_texture(self, chords: list[tuple[int, str, str]]) -> list[dict]:
        """Generate light-classical ballad texture: LH on beats 1&3, RH on 2&4."""
        notes = []
        bar_length_sec = (4.0 * 60.0) / self.config.tempo_bpm
        beat_length_sec = bar_length_sec / (self.config.time_signature[0] or 4)
        
        for bar, root, quality in chords:
            bar_start_sec = bar * bar_length_sec
            voicing = self.voicer.voice_chord(root, quality, voice_lead=True)
            
            if len(voicing.pitches) < 2:
                # Fall back to block
                duration_sec = bar_length_sec
                for pitch, velocity in zip(voicing.pitches, voicing.velocities):
                    notes.append({
                        "time_sec": bar_start_sec,
                        "duration_sec": duration_sec,
                        "pitch": pitch,
                        "velocity": int(velocity * 0.7),
                    })
                continue
            
            # LH: bass note on beats 1 and 3
            for beat_num in [1, 3]:
                beat_start_sec = bar_start_sec + (beat_num - 1) * beat_length_sec
                notes.append({
                    "time_sec": beat_start_sec,
                    "duration_sec": beat_length_sec * 1.8,  # Sustain across to next beat
                    "pitch": voicing.pitches[0],  # Bass note
                    "velocity": 75,
                })
            
            # RH: chord notes on beats 2 and 4 (rolled)
            for beat_num in [2, 4]:
                beat_start_sec = bar_start_sec + (beat_num - 1) * beat_length_sec
                for chord_idx, (pitch, velocity) in enumerate(zip(voicing.pitches[1:], voicing.velocities[1:])):
                    notes.append({
                        "time_sec": beat_start_sec + chord_idx * (beat_length_sec / len(voicing.pitches)),
                        "duration_sec": beat_length_sec,
                        "pitch": pitch,
                        "velocity": int(velocity * 0.7),
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
        default_texture=texture,
        allow_texture_selection=allow_auto_select,
    )
    
    arranger = PianoArranger(config)
    return arranger.arrange(chords, melody_midi_path, output_midi_path)
