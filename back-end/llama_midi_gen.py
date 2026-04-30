"""Generate piano accompaniment MIDI from a lead vocal MIDI using dx2102/llama-midi.

Converts melody to the model's text grid (pitch duration wait velocity instrument),
runs text-generation continuation, parses output with symusic, and writes **piano-only**
tracks to ``chord_gen.mid`` and ``textured_chord_gen.mid`` (same content).
"""
from __future__ import annotations

import logging
import math
import os
import re
import sys
from typing import Any

# Vocal line uses GM program 53 (voice choir) so generations on program 0 stay piano accompaniment.
_MELODY_INSTRUMENT = "53"
_PIANO_INSTRUMENT = "0"

# Lines the model adds after the prompt (same idea as HF Space postprocess).
_NOTE_LINE = re.compile(
    r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\S+)\s*$",
    re.MULTILINE,
)

_pipeline: Any | None = None


def _completion_after_prompt(full_text: str, prompt: str, melody_body: str) -> str:
    """
    HF causal LM returns input + continuation in ``generated_text``. Using the last ``\\n\\n`` block
    fails when the prompt has no blank line; we strip the exact prompt substring when present.
    """
    ft = full_text
    p = prompt
    if p and ft.startswith(p):
        return ft[len(p) :].lstrip("\n\r")
    idx = ft.find(p)
    if idx >= 0:
        return ft[idx + len(p) :].lstrip("\n\r")
    # Decoder may add a BOS wrapper; drop common Llama 3 prefixes then retry.
    for prefix in ("\ufeff", "<|begin_of_text|>"):
        if ft.startswith(prefix):
            ft2 = ft[len(prefix) :].lstrip()
            if p and ft2.startswith(p):
                return ft2[len(p) :].lstrip("\n\r")
            j = ft2.find(p)
            if j >= 0:
                return ft2[j + len(p) :].lstrip("\n\r")
    # Anchor after the melody block we sent (last resort).
    mb = melody_body.strip()
    pos = ft.rfind(mb)
    if pos != -1:
        return ft[pos + len(mb) :].lstrip("\n\r")
    # Last resort: keep text after final header line (model often repeats structure).
    hdr = "pitch duration wait velocity instrument"
    parts = ft.rsplit(hdr, 1)
    if len(parts) > 1:
        tail = parts[-1].lstrip("\n\r")
        if tail.strip():
            return tail
    return ft.split("\n\n")[-1].strip()


def _device_and_dtype():
    """Pick device. Prefers GPU (CUDA/MPS) for much faster generation."""
    import torch

    forced = (os.environ.get("LLAMA_MIDI_DEVICE") or "").strip().lower()
    if forced == "cpu":
        logging.info("LLaMA-MIDI: using CPU (forced via LLAMA_MIDI_DEVICE=cpu)")
        return "cpu", torch.float32
    if forced == "cuda" and torch.cuda.is_available():
        logging.info("LLaMA-MIDI: using CUDA GPU (forced via LLAMA_MIDI_DEVICE=cuda)")
        return "cuda", torch.float16
    if forced == "mps":
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            logging.info("LLaMA-MIDI: using Apple Silicon GPU/MPS (forced via LLAMA_MIDI_DEVICE=mps)")
            logging.info("  → This is MUCH faster than CPU! If generation crashes, set LLAMA_MIDI_DEVICE=cpu")
            return "mps", torch.float16
        logging.warning("LLaMA-MIDI: MPS requested but not available; using CPU")
        return "cpu", torch.float32

    # Auto-detect: prefer GPU when available
    if torch.cuda.is_available():
        logging.info("LLaMA-MIDI: using CUDA GPU (auto-detected, ~10x faster than CPU)")
        return "cuda", torch.float16
    
    # For Apple Silicon: use MPS when LLAMA_MIDI_DEVICE is explicitly set to "mps" or "auto"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        if sys.platform == "darwin" and forced != "auto":
            # Default behavior: warn about MPS but use CPU for safety
            logging.info(
                "LLaMA-MIDI: Apple Silicon GPU detected but using CPU for stability. "
                "Set LLAMA_MIDI_DEVICE=mps to use GPU (~5-10x faster, may crash on very long songs)."
            )
            return "cpu", torch.float32
        else:
            # User explicitly wants MPS via "auto" or we're using it
            logging.info("LLaMA-MIDI: using Apple Silicon GPU/MPS (~5-10x faster than CPU)")
            return "mps", torch.float16
    
    logging.info("LLaMA-MIDI: using CPU (no GPU detected)")
    return "cpu", torch.float32


def _get_pipeline():
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    import torch
    from transformers import pipeline

    model_id = os.environ.get("LLAMA_MIDI_MODEL", "dx2102/llama-midi")
    device, torch_dtype = _device_and_dtype()
    logging.info("LLaMA-MIDI: loading %s on %s (%s)", model_id, device, torch_dtype)
    _pipeline = pipeline(
        "text-generation",
        model=model_id,
        torch_dtype=torch_dtype,
        device=device if device != "mps" else "mps",
        trust_remote_code=False,
    )
    return _pipeline


def melody_midi_to_llama_text(midi_path: str, melody_program: str = _MELODY_INSTRUMENT) -> str:
    """Encode track 0 of MIDI into llama-midi lines (milliseconds, relative waits)."""
    from pretty_midi import PrettyMIDI

    pm = PrettyMIDI(midi_path)
    if not pm.instruments or not pm.instruments[0].notes:
        raise ValueError("Melody MIDI has no notes on track 0")

    notes = sorted(pm.instruments[0].notes, key=lambda n: (n.start if n.start is not None else 0, -n.pitch))
    valid_notes = [n for n in notes if n.start is not None]
    if not valid_notes:
        raise ValueError("Melody MIDI has no valid notes with start times")
    t0 = float(min(n.start for n in valid_notes))
    rows: list[str] = []
    for i, n in enumerate(valid_notes):
        if n.start is None or n.end is None:
            continue
        start_ms = int(round((n.start - t0) * 1000.0))
        dur_ms = max(1, int(round((n.end - n.start) * 1000.0)))
        vel = max(1, min(127, int(n.velocity)))
        if i + 1 < len(valid_notes):
            next_n = valid_notes[i + 1]
            if next_n.start is not None:
                next_start_ms = int(round((next_n.start - t0) * 1000.0))
                wait_ms = max(0, next_start_ms - start_ms)
            else:
                wait_ms = 0
            try:
                tempo = float(pm.estimate_tempo())
            except (TypeError, ValueError):
                tempo = 120.0
            if not math.isfinite(tempo):
                tempo = 120.0
            tempo = max(48.0, min(220.0, tempo))
            quarter_ms = int(round(60000.0 / tempo))
            wait_ms = max(quarter_ms, 300)

        rows.append(f"{int(n.pitch)} {dur_ms} {wait_ms} {vel} {melody_program}")

    return "\n".join(rows)


def llama_text_to_midi_filtered(
    text_block: str,
    out_path: str,
    keep_programs: set[str] | None = None,
    exclude_programs: set[str] | None = None,
) -> int:
    """Parse llama-midi token text into a MIDI file; optionally keep only certain instrument programs."""
    try:
        import symusic
    except ImportError as e:
        raise RuntimeError("symusic is required for LLaMA-MIDI output (pip install symusic)") from e

    body = text_block.strip()
    tracks: dict[str, Any] = {}
    now = 0
    note_count = 0
    header = re.compile(r"^pitch\s+duration\s+wait\s+velocity\s+instrument\s*$", re.I)
    for raw in body.split("\n"):
        line = raw.strip()
        if not line or header.match(line):
            continue
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            pitch, duration, wait, velocity, instrument = parts
            p, d, w, v = int(pitch), int(duration), int(wait), int(velocity)
        except ValueError:
            continue
        inst = str(instrument).strip()
        if keep_programs is not None and inst not in keep_programs:
            now += w
            continue
        if exclude_programs is not None and inst in exclude_programs:
            now += w
            continue
        if inst not in tracks:
            tr = symusic.core.TrackSecond()
            if inst == "drum":
                tr.is_drum = True
            else:
                try:
                    tr.program = int(inst)
                except ValueError:
                    tr.program = 0
            tracks[inst] = tr
        tracks[inst].notes.append(
            symusic.core.NoteSecond(
                time=now / 1000.0,
                duration=d / 1000.0,
                pitch=int(p),
                velocity=max(1, min(127, int(v * 4))),
            )
        )
        note_count += 1
        now += w

    if not tracks:
        raise RuntimeError("LLaMA-MIDI parse produced no notes (empty or invalid continuation)")

    score = symusic.Score(ttype="Second")
    score.tracks.extend(tracks.values())
    score.dump_midi(out_path)
    logging.info("LLaMA-MIDI: wrote %s (%s notes)", out_path, note_count)
    return note_count


def _piano_only_path(completion_text: str, session_dir: str) -> str:
    tmp = os.path.join(session_dir, "llama_midi_raw.mid")
    try:
        llama_text_to_midi_filtered(completion_text, tmp, keep_programs={_PIANO_INSTRUMENT})
        return tmp
    except Exception as e:
        logging.warning("LLaMA-MIDI: no piano program 0 in output (%s); using all except melody", e)
        alt = os.path.join(session_dir, "llama_midi_acc.mid")
        try:
            llama_text_to_midi_filtered(completion_text, alt, exclude_programs={_MELODY_INSTRUMENT})
            return alt
        except Exception as e2:
            raise RuntimeError(
                "LLaMA-MIDI output had no accompaniment notes (only melody/invalid rows)."
            ) from e2


def _note_lines_block(text: str) -> str:
    """Keep only lines that look like llama-midi rows (HF Space postprocess style)."""
    lines = []
    for m in _NOTE_LINE.finditer(text):
        lines.append(m.group(0).strip())
    return "\n".join(lines)


def _has_non_melody_rows(text_block: str) -> bool:
    for raw in text_block.splitlines():
        parts = raw.strip().split()
        if len(parts) != 5:
            continue
        if parts[4] != _MELODY_INSTRUMENT:
            return True
    return False


def _count_unique_pitches_in_llama_text(text_block: str, exclude_program: str = _MELODY_INSTRUMENT) -> int:
    """Count unique pitches in non-melody parts to detect repetitive output."""
    pitches = set()
    for raw in text_block.splitlines():
        parts = raw.strip().split()
        if len(parts) != 5:
            continue
        if parts[4] == exclude_program:  # Skip melody
            continue
        try:
            pitches.add(int(parts[0]))
        except (ValueError, IndexError):
            continue
    return len(pitches)


def _midi_end_seconds(midi_path: str) -> float:
    from pretty_midi import PrettyMIDI

    pm = PrettyMIDI(midi_path)
    end_s = 0.0
    for inst in pm.instruments:
        for n in inst.notes:
            if n.end > end_s:
                end_s = float(n.end)
    return end_s


def _extend_accompaniment_to_target(accompaniment_midi: str, target_seconds: float) -> None:
    """
    If accompaniment is much shorter than target, tile its note content forward in time
    so the generated backing can cover the full vocal length.
    WARNING: Excessive tiling of very short patterns creates repetitive output.
    """
    from pretty_midi import Note, PrettyMIDI

    if target_seconds <= 1.0:
        return

    pm = PrettyMIDI(accompaniment_midi)
    current_end = 0.0
    for inst in pm.instruments:
        for n in inst.notes:
            if n.end > current_end:
                current_end = float(n.end)

    # Only extend when clearly short.
    if current_end >= target_seconds * 0.9:
        return
    if current_end < 0.25:
        return

    # Detect if the generated pattern is suspiciously short (less than 10 seconds)
    # This indicates poor model output that will create looping bars
    if current_end < 10.0:
        raise RuntimeError(
            f"⚠️ LLaMA-MIDI generated VERY SHORT accompaniment ({current_end:.2f}s). "
            f"Tiling this will create a looping bar effect! "
            f"Increase LLAMA_MIDI_MAX_NEW_TOKENS to 20000+ and LLAMA_MIDI_TEMPERATURE to 1.0+. "
            f"The model must generate accompaniment that follows the full melody, not just a repeating pattern."
        )
    
    # Strict tiling limits to prevent looping bar effect
    # Refuse to tile more than 3 times for short patterns
    if current_end < 20.0:
        max_tiles = 2  # Very limited tiling for short patterns
        logging.warning(
            "Pattern is only %.2fs - will tile max %d times. Output may not cover full song. "
            "Generate longer patterns with higher LLAMA_MIDI_MAX_NEW_TOKENS.",
            current_end,
            max_tiles,
        )
    elif current_end < 40.0:
        max_tiles = 4
    else:
        max_tiles = 999
    
    logging.info(
        "LLaMA-MIDI: extending accompaniment from %.2fs to %.2fs by tiling pattern (max %d tiles)",
        current_end,
        target_seconds,
        max_tiles,
    )

    tiles_used = 0
    for inst in pm.instruments:
        base_notes = sorted(inst.notes, key=lambda n: (n.start if n.start is not None else 0, n.pitch))
        if not base_notes:
            continue
        offset = current_end
        while offset < target_seconds and tiles_used < max_tiles:
            for n in base_notes:
                if n.start is None or n.end is None:
                    continue
                ns = n.start + offset
                if ns >= target_seconds:
                    continue
                ne = min(target_seconds, n.end + offset)
                if ne <= ns:
                    continue
                inst.notes.append(
                    Note(
                        velocity=int(n.velocity),
                        pitch=int(n.pitch),
                        start=float(ns),
                        end=float(ne),
                    )
                )
            offset += current_end
            tiles_used += 1

    if tiles_used >= max_tiles:
        logging.warning(
            "⚠️ Reached maximum tile limit (%d) - accompaniment may not cover full song length. "
            "Generated pattern was only %.2fs long.",
            max_tiles,
            current_end,
        )

    pm.write(accompaniment_midi)


def _analyze_melody_chords(midi_path: str, tonic: str, mode: str) -> list[str]:
    """
    Analyze melody MIDI to extract rough chord progression.
    Returns list of chord symbols that the melody implies.
    """
    try:
        from pretty_midi import PrettyMIDI
        
        pm = PrettyMIDI(midi_path)
        if not pm.instruments or not pm.instruments[0].notes:
            return ["I", "V", "vi", "IV"]  # Default progression
        
        # Get tonic pitch number
        tonic_map = {
            "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
            "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
            "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
        }
        tonic_pitch = tonic_map.get(tonic, 0)
        
        # Major/minor scale degrees to chord quality
        if mode == "maj":
            # Major scale: I ii iii IV V vi vii°
            scale_chords = ["I", "ii", "iii", "IV", "V", "vi", "vii°"]
            scale_pitches = [0, 2, 4, 5, 7, 9, 11]  # Major scale intervals
        else:
            # Minor scale: i ii° III iv v VI VII
            scale_chords = ["i", "ii°", "III", "iv", "v", "VI", "VII"]
            scale_pitches = [0, 2, 3, 5, 7, 8, 10]  # Natural minor intervals
        
        # Analyze melody in 2-bar segments to detect chord changes
        notes = sorted(pm.instruments[0].notes, key=lambda n: n.start)
        if not notes:
            return ["I", "V", "vi", "IV"]
        
        duration = notes[-1].end
        bar_length = 2.0  # Assume 2-bar chord changes at 120 BPM
        num_segments = max(1, int(duration / bar_length))
        
        chord_progression = []
        for seg in range(min(num_segments, 8)):  # Max 8 segments (16 bars)
            seg_start = seg * bar_length
            seg_end = (seg + 1) * bar_length
            seg_notes = [n for n in notes if seg_start <= n.start < seg_end]
            
            if not seg_notes:
                # Copy previous chord or use tonic
                chord_progression.append(chord_progression[-1] if chord_progression else scale_chords[0])
                continue
            
            # Count pitch classes in this segment
            pitch_classes = [0] * 12
            for n in seg_notes:
                pc = (n.pitch - tonic_pitch) % 12
                pitch_classes[pc] += 1
            
            # Find which scale degree has the most notes
            best_degree = 0
            best_score = -1
            for i, scale_pc in enumerate(scale_pitches):
                # Score based on root, third, fifth presence
                third = scale_pitches[(i + 2) % len(scale_pitches)]
                fifth = scale_pitches[(i + 4) % len(scale_pitches)]
                
                score = pitch_classes[scale_pc] * 3  # Root is most important
                score += pitch_classes[third] * 2    # Third defines quality
                score += pitch_classes[fifth] * 1    # Fifth adds stability
                
                if score > best_score:
                    best_score = score
                    best_degree = i
            
            chord_progression.append(scale_chords[best_degree])
        
        # If too short, extend with common patterns
        if len(chord_progression) < 4:
            # Use common Cantonese ballad progression: I-V-vi-IV or i-VI-III-VII
            if mode == "maj":
                default = ["I", "V", "vi", "IV"]
            else:
                default = ["i", "VI", "III", "VII"]
            while len(chord_progression) < 4:
                chord_progression.append(default[len(chord_progression) % 4])
        
        logging.info("Analyzed melody chord progression: %s", " → ".join(chord_progression[:8]))
        return chord_progression
        
    except Exception as e:
        logging.warning("Melody chord analysis failed: %s", e)
        # Return common Cantonese ballad progression
        if mode == "maj":
            return ["I", "V", "vi", "IV", "I", "V", "vi", "IV"]
        else:
            return ["i", "VI", "III", "VII", "i", "VI", "III", "VII"]


def _generate_intro(tonic: str, mode: str, pickup_bars: float = 2.0) -> list[str]:
    """
    Generate piano intro before vocal enters.
    Uses arpeggios and melodic fragments in the song's key.
    
    Args:
        tonic: Key tonic (C, D, Eb, etc.)
        mode: 'maj' or 'min'
        pickup_bars: Duration in bars (default 2 bars intro)
    
    Returns:
        List of llama-midi lines for the intro
    """
    tonic_map = {
        "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
        "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
        "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
    }
    offset = tonic_map.get(tonic, 0)
    
    # Create intro: 2 bars = 8 beats = 32 sixteenth notes
    # Use I-V progression for intro in major, i-v in minor
    if mode == "maj":
        # Major intro: I → V (uplifting)
        # Bar 1: Tonic chord arpeggio (soft intro)
        intro = [
            (36, 1920, 0, 12),     # C1 (bass - sustained)
            (48, 480, 480, 11),    # C2 (quarter note)
            (60, 360, 240, 14),    # C4 (dotted 8th - arpeggio starts)
            (64, 360, 240, 13),    # E4
            (67, 360, 240, 15),    # G4
            (72, 360, 240, 14),    # C5
            (71, 720, 480, 16),    # B4 (leading tone - anticipate V)
            # Bar 2: V chord (builds anticipation for vocal entry)
            (43, 1920, 0, 13),     # G1 (bass)
            (55, 480, 480, 12),    # G2
            (62, 360, 240, 15),    # D4 (arpeggio)
            (67, 360, 240, 14),    # G4
            (71, 360, 240, 17),    # B4
            (74, 480, 240, 16),    # D5 (peak)
            (71, 480, 480, 14),    # B4 (descend toward vocal entry)
            (67, 480, 960, 13),    # G4 (prepare for vocal - longer wait)
        ]
    else:
        # Minor intro: i → V (melancholic then tension)
        intro = [
            (36, 1920, 0, 12),     # C1 (minor tonic bass)
            (48, 480, 480, 11),    # C2
            (60, 360, 240, 14),    # C4 (arpeggio)
            (63, 360, 240, 13),    # Eb4 (minor 3rd - sets mood)
            (67, 360, 240, 15),    # G4
            (72, 360, 240, 14),    # C5
            (70, 720, 480, 16),    # Bb4 (minor 7th)
            # Bar 2: V (dominant - builds tension)
            (43, 1920, 0, 13),     # G1
            (55, 480, 480, 12),    # G2
            (62, 360, 240, 15),    # D4
            (65, 360, 240, 14),    # F4 (7th for tension)
            (71, 360, 240, 17),    # B4 (leading tone)
            (74, 480, 240, 16),    # D5
            (71, 480, 480, 14),    # B4
            (67, 480, 960, 13),    # G4 (resolve to vocal)
        ]
    
    # Transpose to key
    transposed = []
    for pitch, dur, wait, vel in intro:
        new_pitch = pitch + offset
        transposed.append(f"{new_pitch} {dur} {wait} {vel} 0")
    
    logging.info("Generated %d-bar intro in %s %s", int(pickup_bars), tonic, "major" if mode == "maj" else "minor")
    return transposed


def _transpose_seed_to_key(tonic: str, mode: str) -> list[str]:
    """
    Generate ballad-style piano seed pattern transposed to the given key.
    Uses COMPLEMENTARY RHYTHM with TWO textures:
    1. Sustained long chords for soft (pp/mp) vocal sections
    2. Flowing arpeggios for medium sections / when melody rests
    Returns list of llama-midi text lines.
    """
    # Map tonic names to semitone offset from C
    tonic_map = {
        "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
        "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
        "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
    }
    
    offset = tonic_map.get(tonic, 0)
    
    # Base pattern in C major with MIXED TEXTURES AND BEAT EMPHASIS
    # Format: (pitch_offset_from_C, duration, wait, velocity)
    # Shows: SUSTAINED CHORDS (for soft vocals), ARPEGGIOS (for fills), BEAT 1 EMPHASIS
    if mode == "maj":
        # Major key ballad: Demonstrates sustained, flowing textures, AND beat 1 emphasis
        base_pattern = [
            # Bar 1 DOWNBEAT: Imaj7 (Cmaj7) - SUSTAINED CHORD with STRONG beat 1
            # Beat 1 = STRONGER (vel 18-20), other beats = softer (12-14)
            (36, 3840, 0, 20),     # C1 (STRONG on beat 1 - downbeat emphasis!)
            (48, 3840, 0, 19),     # C2 (STRONG on beat 1)
            (60, 3840, 0, 18),     # C4 (STRONG on beat 1 - sets the bar)
            (64, 3840, 0, 12),     # E4 (softer - not on beat 1)
            (67, 3840, 0, 14),     # G4 (softer)
            (71, 3840, 960, 13),   # B4 (maj7 - soft and sustained)
            
            # Bar 2 DOWNBEAT: V (G) - TRANSITION with beat 1 emphasis + gentle arpeggio
            (43, 2880, 0, 19),     # G1 (STRONG beat 1 - new bar!)
            (55, 2880, 0, 18),     # G2 (STRONG beat 1)
            (62, 480, 240, 14),    # D4 (softer - beat 2)
            (67, 480, 240, 13),    # G4 (softer)
            (71, 480, 240, 15),    # B4 (softer)
            (74, 960, 240, 14),    # D5 (hold)
            (71, 480, 240, 12),    # B4 (gentle)
            
            # Bar 3 DOWNBEAT: vi (Am) - SUSTAINED CHORD with beat 1 emphasis
            (33, 3840, 0, 20),     # A0 (STRONG beat 1 - new bar!)
            (45, 3840, 0, 19),     # A1 (STRONG)
            (57, 3840, 0, 18),     # A3 (STRONG beat 1)
            (60, 3840, 0, 12),     # C4 (softer)
            (64, 3840, 0, 14),     # E4 (softer)
            (69, 3840, 960, 13),   # A4 (soft)
            
            # Bar 4 DOWNBEAT: IV (F) - FLOWING ARPEGGIOS with beat 1 accent
            (41, 2880, 0, 20),     # F1 (STRONG beat 1 - sustained bass)
            (53, 240, 240, 18),    # F2 (STRONG on beat 1 - short note)
            (60, 240, 240, 14),    # C4 (softer - off beat)
            (65, 240, 240, 13),    # F4 (softer)
            (69, 240, 240, 16),    # A4 (slightly louder - peak)
            (72, 240, 240, 15),    # C5
            (76, 240, 240, 14),    # E5 (maj7)
            (72, 240, 240, 13),    # C5 (descend)
            (69, 240, 240, 12),    # A4
            (65, 480, 480, 11),    # F4 (settle)
            
            # Bar 5 DOWNBEAT: I (C) - RETURN with strongest beat 1 (climactic)
            (36, 4800, 0, 22),     # C1 (STRONGEST beat 1 - final cadence!)
            (48, 4800, 0, 21),     # C2 (very strong)
            (60, 4800, 0, 20),     # C4 (strong tonic return)
            (64, 4800, 0, 13),     # E4 (softer)
            (67, 4800, 0, 15),     # G4 (softer)
            (71, 4800, 0, 14),     # B4 (maj7)
            (72, 4800, 1920, 13),  # C5 (long hold, then silence)
        ]
    else:  # minor mode
        # Minor key: Same principle - beat 1 emphasis + mix of sustained/arpeggios
        base_pattern = [
            # Bar 1 DOWNBEAT: i (Cm) - SUSTAINED MINOR CHORD with beat 1 emphasis
            (36, 3840, 0, 20),     # C1 (STRONG beat 1)
            (48, 3840, 0, 19),     # C2 (STRONG)
            (60, 3840, 0, 18),     # C4 (STRONG - sets downbeat)
            (63, 3840, 0, 12),     # Eb4 (minor 3rd - softer)
            (67, 3840, 0, 14),     # G4 (softer)
            (70, 3840, 960, 13),   # Bb4 (minor 7th)
            
            # Bar 2 DOWNBEAT: V7 (G7) - SUSTAINED with beat 1 accent
            (43, 2880, 0, 19),     # G1 (STRONG beat 1)
            (55, 2880, 0, 18),     # G2 (STRONG)
            (62, 480, 240, 14),    # D4 (gentle arpeggio - softer)
            (65, 480, 240, 13),    # F4 (7th)
            (71, 480, 240, 15),    # B4
            (74, 960, 240, 14),    # D5
            (71, 480, 240, 12),    # B4
            
            # Bar 3 DOWNBEAT: VI (Ab) - SUSTAINED MAJOR CHORD with warmth
            (32, 3840, 0, 20),     # Ab0 (STRONG beat 1)
            (44, 3840, 0, 19),     # Ab1 (STRONG)
            (56, 3840, 0, 18),     # Ab3 (STRONG)
            (60, 3840, 0, 12),     # C4 (3rd - softer)
            (63, 3840, 0, 14),     # Eb4 (softer)
            (68, 3840, 960, 13),   # Ab4
            
            # Bar 4 DOWNBEAT: iv (Fm) - FLOWING ARPEGGIO with beat 1 accent
            (41, 2880, 0, 20),     # F1 (STRONG beat 1)
            (53, 240, 240, 18),    # F2 (STRONG - 8th note)
            (60, 240, 240, 14),    # C4 (softer)
            (65, 240, 240, 13),    # F4 (softer)
            (68, 240, 240, 16),    # Ab4 (minor 3rd - peak)
            (72, 240, 240, 15),    # C5
            (75, 240, 240, 14),    # Eb5
            (72, 240, 240, 13),    # C5
            (68, 240, 240, 12),    # Ab4
            (65, 480, 480, 11),    # F4
            
            # Bar 5 DOWNBEAT: i (Cm) - SUSTAINED MINOR TONIC (strong return)
            (36, 4800, 0, 22),     # C1 (STRONGEST - minor cadence)
            (48, 4800, 0, 21),     # C2 (very strong)
            (60, 4800, 0, 20),     # C4 (strong tonic)
            (63, 4800, 0, 13),     # Eb4 (minor - softer)
            (67, 4800, 0, 15),     # G4 (softer)
            (70, 4800, 0, 14),     # Bb4
            (72, 4800, 1920, 13),  # C5 (long hold)
        ]
    
    # Transpose each pitch by the offset
    transposed = []
    for pitch, dur, wait, vel in base_pattern:
        new_pitch = pitch + offset
        transposed.append(f"{new_pitch} {dur} {wait} {vel} 0")
    
    return transposed


def generate_accompaniment(
    midi_path: str,
    out_dir: str,
    *,
    title: str = "Lead vocal",
    tempo: float = 120.0,
    tonic: str = "C",
    mode: str = "maj",
    pickup_shift: int = 0,
    beat_times_sec: list[float] | None = None,
    beat_numbers: list[int] | None = None,
) -> None:
    """Read vocal/lead MIDI, run LLaMA-MIDI, write ``chord_gen.mid`` and ``textured_chord_gen.mid``.
    
    Args:
        midi_path: Path to vocal melody MIDI
        out_dir: Output directory
        title: Song title/description for generation
        tempo: Tempo in BPM
        tonic: Key tonic (C, D, Eb, etc.)
        mode: 'maj' or 'min' 
        pickup_shift: Number of 16th notes before first downbeat (0 if starts on beat 1)
    """
    pipe = _get_pipeline()

    # Log what key we're generating in
    logging.info("=" * 60)
    logging.info("Generating piano accompaniment in: %s %s", tonic, "major" if mode == "maj" else "minor")
    logging.info("Tempo: %.1f BPM", tempo)
    if pickup_shift > 0:
        logging.info("Pickup: %d sixteenth notes", pickup_shift)
    if beat_times_sec and len(beat_times_sec) > 0:
        logging.info("Beat grid: %d beats from original song", len(beat_times_sec))
    logging.info("=" * 60)

    melody_body = melody_midi_to_llama_text(midi_path)
    header = "pitch duration wait velocity instrument"
    # HF Space: if the prefix is a single-line title, append "\\n" so the model does not "continue" the title.
    title_line = title.strip()
    if title_line and "\n" not in title_line:
        title_line = title_line + "\n"
    
    # Analyze beat grid from original song for rhythm matching
    rhythm_instruction = ""
    if beat_times_sec and len(beat_times_sec) > 4:
        # Calculate actual tempo from beat grid (more accurate than estimates)
        beat_intervals = [beat_times_sec[i+1] - beat_times_sec[i] 
                         for i in range(min(8, len(beat_times_sec) - 1))]
        avg_beat_sec = sum(beat_intervals) / len(beat_intervals) if beat_intervals else 0.5
        actual_bpm = 60.0 / avg_beat_sec if avg_beat_sec > 0.1 else tempo
        
        # Determine time signature from beat numbers if available
        time_sig = "4/4"
        if beat_numbers and len(beat_numbers) > 4:
            max_beat = max(beat_numbers[:16])
            if max_beat == 3:
                time_sig = "3/4"
            elif max_beat == 6:
                time_sig = "6/8"
        
        rhythm_instruction = (
            f"# ORIGINAL SONG RHYTHM ANALYSIS:\n"
            f"# - Tempo: {actual_bpm:.1f} BPM (from audio beat tracking)\n"
            f"# - Time signature: {time_sig}\n"
            f"# - Beat grid: {len(beat_times_sec)} beats detected\n"
            f"# ⚠️ MATCH THIS RHYTHM EXACTLY - align chords to these beat timings\n"
        )
        logging.info("Beat grid: %d beats, avg tempo %.1f BPM, time sig %s", 
                    len(beat_times_sec), actual_bpm, time_sig)
    
    # Analyze melody to extract chord progression
    chord_progression = _analyze_melody_chords(midi_path, tonic, mode)
    chord_str = " → ".join(chord_progression[:8])
    
    # Generate intro if song has pickup (vocal doesn't start on beat 1)
    intro_section = ""
    if pickup_shift > 0:
        intro_bars = pickup_shift / 16.0  # Convert 16th notes to bars
        if intro_bars < 1.0:
            intro_bars = 2.0  # Always at least 2 bars intro
        intro_lines = _generate_intro(tonic, mode, intro_bars)
        intro_section = "\n".join(intro_lines) + "\n"
        logging.info("Added %.1f-bar intro before vocal (pickup_shift=%d)", intro_bars, pickup_shift)
    
    # Generate ballad-style piano seed transposed to the song's key
    # Note: velocities will be multiplied by 4 in MIDI output, so 14-18 here = 56-72 final.
    # These rows are part of the prompt only (not parsed as output).
    seed_piano_lines = _transpose_seed_to_key(tonic, mode)
    seed_piano = "\n".join(seed_piano_lines)
    
    # Add instructional comment to guide the model to follow the melody harmonically
    instruction = (
        "# ============================================\n"
        "# TASK: Generate piano accompaniment (program 0) for the vocal melody above\n"
        "# ============================================\n"
        f"{rhythm_instruction}"
        "# Piano accompaniment guidelines (CRITICAL):\n"
        "# 1. Use instrument program 0 (piano) - NOT program 53 (melody)\n"
        "# 2. MATCH ORIGINAL RHYTHM - analyze melody timing and align piano accordingly\n"
        "# 3. COMPLEMENTARY RHYTHM - fill gaps between melody notes, don't double melody rhythm\n"
        "# 4. For SOFT vocals: sustained chords (whole notes/half notes)\n"
        "# 5. For MEDIUM vocals: flowing arpeggios matching the song's tempo\n"
        "# 6. When melody RESTS: piano fills and runs\n"
        f"# 7. Follow chord progression: {chord_str}\n"
        "# 8. BEAT 1 EMPHASIS: velocity 18-22 on downbeats, 12-16 otherwise\n"
        "# 9. Change chords at musically appropriate times (bar boundaries, phrase changes)\n"
        "# 10. Continue until melody ends - develop the pattern, don't loop same bar\n"
        "# ============================================\n"
        "#\n"
    )
    
    logging.info("LLaMA-MIDI: seed pattern in %s %s (transposed %s semitones)", 
                 tonic, "major" if mode == "maj" else "minor",
                 {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5, 
                  "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11}.get(tonic, 0))
    
    # Build prompt: title → header → MELODY FIRST → separator → intro → seed piano
    # This structure ensures the model generates piano (program 0), not more melody (program 53)
    piano_header = "# Piano accompaniment (program 0) starts here:\n"
    prompt = f"{title_line}{header}\n{melody_body}\n{instruction}{piano_header}{intro_section}{seed_piano}\n".strip()

    max_new = int(os.environ.get("LLAMA_MIDI_MAX_NEW_TOKENS", "16000"))
    temperature = float(os.environ.get("LLAMA_MIDI_TEMPERATURE", "1.0"))
    top_p = float(os.environ.get("LLAMA_MIDI_TOP_P", "0.92"))

    logging.info(
        "LLaMA-MIDI: generating (max_new_tokens=%s, temp=%s, top_p=%s)",
        max_new,
        temperature,
        top_p,
    )

    def _extract_generated_text(result_obj: Any) -> str:
        if isinstance(result_obj, list) and result_obj:
            first = result_obj[0]
            if isinstance(first, dict):
                # Transformers variants have used either generated_text or text keys.
                return str(first.get("generated_text") or first.get("text") or "")
            return str(first)
        if isinstance(result_obj, dict):
            return str(result_obj.get("generated_text") or result_obj.get("text") or "")
        return str(result_obj)

    attempts = [
        {"do_sample": True, "temperature": temperature, "top_p": top_p},
        {"do_sample": True, "temperature": min(1.0, temperature * 1.2), "top_p": min(1.0, top_p + 0.05)},
        {"do_sample": True, "temperature": min(1.2, temperature * 1.5), "top_p": 0.92},
        {"do_sample": False},
    ]
    full_text = ""
    completion = ""
    completion_notes = ""
    for idx, params in enumerate(attempts, start=1):
        logging.info("LLaMA-MIDI: generation attempt %s/%s params=%s", idx, len(attempts), params)
        
        # Clear GPU memory before generation to prevent OOM
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                # MPS memory cleanup
                torch.mps.empty_cache()
        except Exception:
            pass
        
        result = pipe(prompt, max_new_tokens=max_new, **params)
        full_text = _extract_generated_text(result)
        completion = _completion_after_prompt(full_text, prompt, melody_body)
        completion_notes = _note_lines_block(completion)
        if not completion_notes.strip():
            completion_notes = _note_lines_block(full_text)
        
        # Check if we have non-melody content
        if not (completion_notes.strip() and _has_non_melody_rows(completion_notes)):
            if completion_notes.strip() and not _has_non_melody_rows(completion_notes):
                logging.warning(
                    "LLaMA-MIDI: attempt %s produced only melody program %s rows; retrying",
                    idx,
                    _MELODY_INSTRUMENT,
                )
            else:
                logging.warning(
                    "LLaMA-MIDI: attempt %s produced no note-like lines (completion first 400 chars: %r)",
                    idx,
                    completion[:400],
                )
            continue
        
        # Check for repetitive output (e.g., only 2-3 unique pitches)
        unique_pitches = _count_unique_pitches_in_llama_text(completion_notes)
        note_lines = len([line for line in completion_notes.splitlines() if line.strip()])
        
        # Stricter quality checks to prevent looping patterns
        if unique_pitches < 6 and note_lines > 10:
            logging.warning(
                "LLaMA-MIDI: attempt %s produced VERY REPETITIVE output "
                "(only %d unique pitches in %d notes); retrying with higher temperature",
                idx,
                unique_pitches,
                note_lines,
            )
            continue
        
        # Require minimum output length to prevent short loops
        if note_lines < 100:
            logging.warning(
                "LLaMA-MIDI: attempt %s produced TOO SHORT output "
                "(only %d note lines, need 100+); retrying for longer generation",
                idx,
                note_lines,
            )
            continue
        
        if unique_pitches >= 6:
            logging.info(
                "LLaMA-MIDI: attempt %s succeeded with %d unique pitches in %d note lines",
                idx,
                unique_pitches,
                note_lines,
            )
            break

    raw_out = os.path.join(out_dir, "llama_raw_output.txt")
    try:
        with open(raw_out, "w", encoding="utf-8") as f:
            f.write(full_text or "")
    except OSError:
        pass

    if not completion_notes.strip():
        raise RuntimeError(
            "LLaMA-MIDI produced no usable accompaniment lines after retries. "
            f"See {raw_out} for raw model output."
        )
    
    # Final check for quality - stricter to prevent looping bars
    unique_pitches = _count_unique_pitches_in_llama_text(completion_notes)
    note_lines = len([line for line in completion_notes.splitlines() if line.strip()])
    
    if note_lines < 100:
        raise RuntimeError(
            f"LLaMA-MIDI generated TOO SHORT output ({note_lines} notes). "
            f"This will create a looping bar effect. Increase LLAMA_MIDI_MAX_NEW_TOKENS to 16000+."
        )
    
    if unique_pitches < 6:
        raise RuntimeError(
            f"LLaMA-MIDI produced POOR QUALITY output: only {unique_pitches} unique pitches in {note_lines} notes. "
            "Output will sound very repetitive (looping same bar). Try: "
            "(1) Increase LLAMA_MIDI_TEMPERATURE to 1.0-1.2, "
            "(2) Increase LLAMA_MIDI_MAX_NEW_TOKENS to 20000+."
        )
    elif unique_pitches < 10:
        logging.warning(
            "LLaMA-MIDI: Limited diversity - only %d unique pitches generated. "
            "May sound somewhat repetitive. Consider increasing temperature to 1.0+.",
            unique_pitches,
        )
    else:
        logging.info("✓ LLaMA-MIDI: Good diversity with %d unique pitches in %d notes", unique_pitches, note_lines)

    # Free memory after generation
    try:
        import torch
        import gc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()
        logging.info("LLaMA-MIDI: cleared GPU/system memory after generation")
    except Exception:
        pass
    
    piano_mid = _piano_only_path(completion_notes, out_dir)
    try:
        target_len = _midi_end_seconds(midi_path)
        _extend_accompaniment_to_target(piano_mid, target_len)
    except Exception as e:
        logging.warning("LLaMA-MIDI: duration extension skipped (%s)", e)
    chord_out = os.path.join(out_dir, "chord_gen.mid")
    textured_out = os.path.join(out_dir, "textured_chord_gen.mid")

    import shutil

    shutil.copyfile(piano_mid, chord_out)
    shutil.copyfile(piano_mid, textured_out)

    log_path = os.path.join(out_dir, "chord_gen_log.json")
    try:
        import json

        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "engine": "llama-midi",
                    "model": os.environ.get("LLAMA_MIDI_MODEL", "dx2102/llama-midi"),
                    "title": title,
                    "tempo_hint": tempo,
                    "prompt_chars": len(prompt),
                    "output_chars": len(full_text),
                    "completion_chars": len(completion),
                    "completion_note_lines": len(completion_notes.splitlines()) if completion_notes else 0,
                    "raw_output_path": raw_out,
                },
                f,
                indent=2,
            )
    except OSError:
        pass
