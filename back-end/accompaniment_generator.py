"""Accompaniment generation orchestration.

Routes between the old LLaMA-MIDI engine and the new rules-based (chord recognition +
piano arranger) engine based on environment variable ACCOMP_ENGINE.
"""
from __future__ import annotations

import logging
import os
from typing import Any


def generate_accompaniment(
    midi_path: str,
    out_dir: str,
    *,
    title: str = "Lead vocal",
    tempo: float = 120.0,
    tonic: str | None = None,
    mode: str | None = None,
    pickup_shift: int = 0,
    beat_times_sec: list[float] | None = None,
    beat_numbers: list[int] | None = None,
) -> dict[str, Any]:
    """
    Generate piano accompaniment using the configured engine.
    
    Args:
        midi_path: Path to vocal melody MIDI.
        out_dir: Output directory.
        title: Song title for generation context.
        tempo: Tempo in BPM.
        tonic: Detected key root ('C', 'G#', etc.).
        mode: 'maj' or 'min'.
        pickup_shift: Sixteenth-note offset for alignment.
        beat_times_sec: Beat grid times.
        beat_numbers: Beat positions in bar.
        
    Returns:
        Dict with keys: 'engine', 'output_files', 'metadata'.
    """
    engine = (os.environ.get("ACCOMP_ENGINE", "rules") or "rules").strip().lower()
    
    if engine == "llama":
        return _generate_with_llama_midi(
            midi_path, out_dir, title=title, tempo=tempo,
            tonic=tonic, mode=mode, pickup_shift=pickup_shift,
            beat_times_sec=beat_times_sec, beat_numbers=beat_numbers
        )
    elif engine == "rules" or engine == "arranger":
        return _generate_with_rules(
            midi_path, out_dir, title=title, tempo=tempo,
            tonic=tonic, mode=mode, pickup_shift=pickup_shift,
            beat_times_sec=beat_times_sec, beat_numbers=beat_numbers
        )
    else:
        logging.error("Unknown ACCOMP_ENGINE=%s; defaulting to 'rules'", engine)
        return _generate_with_rules(
            midi_path, out_dir, title=title, tempo=tempo,
            tonic=tonic, mode=mode, pickup_shift=pickup_shift,
            beat_times_sec=beat_times_sec, beat_numbers=beat_numbers
        )


def _generate_with_llama_midi(
    midi_path: str,
    out_dir: str,
    *,
    title: str,
    tempo: float,
    tonic: str | None,
    mode: str | None,
    pickup_shift: int,
    beat_times_sec: list[float] | None,
    beat_numbers: list[int] | None,
) -> dict[str, Any]:
    """Delegate to the LLaMA-MIDI engine."""
    try:
        import llama_midi_gen
        
        llama_midi_gen.generate_accompaniment(
            midi_path, out_dir,
            title=title,
            tempo=tempo,
            tonic=tonic or "C",
            mode=mode or "maj",
            pickup_shift=pickup_shift or 0,
            beat_times_sec=beat_times_sec or [],
            beat_numbers=beat_numbers or [],
        )
        
        return {
            "engine": "llama-midi",
            "output_files": ["chord_gen.mid", "textured_chord_gen.mid"],
            "metadata": {
                "model": os.environ.get("LLAMA_MIDI_MODEL", "dx2102/llama-midi"),
                "max_tokens": int(os.environ.get("LLAMA_MIDI_MAX_NEW_TOKENS", "12000")),
                "temperature": float(os.environ.get("LLAMA_MIDI_TEMPERATURE", "0.75")),
            }
        }
    except Exception as e:
        logging.exception("LLaMA-MIDI generation failed")
        raise


def _generate_with_rules(
    midi_path: str,
    out_dir: str,
    *,
    title: str,
    tempo: float,
    tonic: str | None,
    mode: str | None,
    pickup_shift: int,
    beat_times_sec: list[float] | None,
    beat_numbers: list[int] | None,
) -> dict[str, Any]:
    """Generate using chord recognition + piano arranger."""
    try:
        import chord_recognition
        import melody_to_chord
        import piano_arranger
        import pretty_midi
        import shutil
    except ImportError as e:
        logging.error("Required module not found: %s", e)
        raise
    
    try:
        tonic = tonic or "C"
        mode = mode or "maj"
        beat_times_sec = beat_times_sec or []
        beat_numbers = beat_numbers or []
        
        logging.info("🎹 Arranging with rules engine: %s %s @ %.1f BPM",
                     tonic, ("major" if mode == "maj" else "minor"), tempo)
        
        # Step 1: Try to recognize chords from accompaniment stem
        # (For now, we'll skip this and use melody-based harmonisation instead)
        # Future: parse beat_times_sec to find accompaniment stem path
        chords = []
        
        # Step 2: Fallback to melody-based harmonisation
        if not chords:
            logging.info("Harmonising from melody MIDI...")
            # Tempo-aware segmentation: slower ballads use 1-bar chords for richer harmony
            bars_per_chord = 1 if tempo < 90 else 2
            logging.info("🎵 Tempo %.1f BPM → %d-bar chord segments", tempo, bars_per_chord)

            harmonised = melody_to_chord.estimate_chords_from_melody(
                midi_path,
                tonic=tonic,
                mode=mode,
                bars_per_chord=bars_per_chord,
            )
            chords = [(c.bar, c.root, c.quality) for c in harmonised]
        
        if not chords:
            logging.warning("Chord estimation failed; using default progression")
            chords = [(0, tonic, mode), (2, "G" if tonic == "C" else tonic, mode),
                     (4, "A" if tonic == "C" else tonic, "min"), (6, "F" if tonic == "C" else tonic, mode)]
        
        # Step 3: Arrange chords into piano MIDI
        logging.info("Arranging %d chords...", len(chords))

        # Select texture based on tempo: ballad texture for slow songs (< 90 BPM)
        if tempo < 90:
            texture = piano_arranger.Texture.BALLAD
            logging.info("🎼 Using BALLAD texture (slow tempo: %.1f BPM)", tempo)
        else:
            texture = piano_arranger.Texture.ARPEGGIO
            logging.info("🎼 Using ARPEGGIO texture (moderate tempo: %.1f BPM)", tempo)

        block_notes, textured_notes = piano_arranger.arrange_piano_accompaniment(
            chords=chords,
            tempo_bpm=tempo,
            time_signature=(4, 4),
            beat_grid_times_sec=beat_times_sec,
            beat_numbers=beat_numbers,
            melody_midi_path=midi_path,
            output_midi_path=None,  # Don't write yet
            texture=texture,
        )
        
        # Step 4: Write both outputs
        _write_notes_to_midi(
            out_dir, "chord_gen.mid",
            block_notes, tempo
        )
        _write_notes_to_midi(
            out_dir, "textured_chord_gen.mid",
            textured_notes, tempo
        )
        
        logging.info("✓ Arranged accompaniment: %d block notes, %d textured notes",
                     len(block_notes), len(textured_notes))
        
        return {
            "engine": "rules-based-arranger",
            "output_files": ["chord_gen.mid", "textured_chord_gen.mid"],
            "metadata": {
                "chords_count": len(chords),
                "block_notes": len(block_notes),
                "textured_notes": len(textured_notes),
                "tonic": tonic,
                "mode": mode,
            }
        }
    
    except Exception as e:
        logging.exception("Rules-based arrangement failed")
        raise


def _write_notes_to_midi(out_dir: str, filename: str, notes: list[dict], tempo_bpm: float) -> None:
    """Write a list of note dicts to a MIDI file."""
    try:
        import pretty_midi
        import os
    except ImportError:
        logging.error("pretty_midi not installed")
        return
    
    try:
        pm = pretty_midi.PrettyMIDI(initial_tempo=tempo_bpm)
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
        
        out_path = os.path.join(out_dir, filename)
        pm.write(out_path)
        logging.info("Wrote %s (%d notes)", filename, len(notes))
    
    except Exception as e:
        logging.error("Failed to write %s: %s", filename, e)
