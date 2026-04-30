"""Infer phrase list and coarse key/tempo from a lead MIDI (e.g. Basic Pitch).

When ``key_hints`` includes ``beat_bpm`` and ``beat_times_sec`` from the instrumental
stem (see youtube_melody._beat_track_no_vocals), tempo is blended with the MIDI map and
phrase lengths are chosen with a small DP that rewards cuts near estimated downbeats.
"""
from __future__ import annotations

import io
import logging
import math
import os
import tempfile
from typing import Any

import numpy as np
from pretty_midi import PrettyMIDI

from key_maps import major_map, minor_map, str_to_root

ALLOWED_PHRASE_BARS = (8, 4)
_PHRASE_LABELS = ("A", "B", "C", "D")

_KK_MAJOR = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88], dtype=np.float64
)
_KK_MINOR = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17], dtype=np.float64
)
_TONIC_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _quantize_note(time: float, unit: float) -> int:
    base = int(time // unit)
    return base if time % unit < unit / 2 else base + 1


def _construct_melo_sequence(all_notes_and_pos: list) -> list:
    def fix_end(max_end: int) -> int:
        return int(((max_end // 4) + 1) * 4)

    def is_note_playing_at_cursor(note, cursor: int) -> bool:
        return note[0] <= cursor < note[1]

    max_end = max(all_notes_and_pos, key=lambda x: x[1])[1]
    fixed_end = fix_end(max_end // 16)
    fixed_end *= 16

    note_dict = {i: all_notes_and_pos[i] for i in range(len(all_notes_and_pos))}
    melo_sequence, cache = [], -1
    for cursor in range(fixed_end):
        if cache != -1:
            if is_note_playing_at_cursor(note_dict[cache], cursor):
                melo_sequence.append(note_dict[cache][2])
                continue
            cache = -1
        for key, note in note_dict.items():
            if is_note_playing_at_cursor(note, cursor):
                melo_sequence.append(note[2])
                cache = key
                break
        else:
            melo_sequence.append(0)
    return melo_sequence


def _pitch_to_number(pitch: int, meta: dict) -> float:
    tonic_distance = str_to_root[meta["tonic"]]
    if meta["mode"] == "maj":
        return major_map[(pitch - tonic_distance) % 12]
    return minor_map[(pitch - tonic_distance) % 12]


def _full_melo_sequence(midi_path: str, meta: dict, note_shift: int = 0) -> tuple[list, float]:
    midi = PrettyMIDI(midi_path)
    _times, tempos = midi.get_tempo_changes()
    tempo = float(tempos[0]) if len(tempos) else 120.0
    meta = {**meta, "tempo": tempo}
    unit = 60.0 / meta["tempo"] / 4.0

    if not midi.instruments or not midi.instruments[0].notes:
        raise ValueError("MIDI has no notes on track 0 (melody).")

    all_notes_and_pos = []
    for note in midi.instruments[0].notes:
        if _quantize_note(note.end, unit) >= note_shift:
            all_notes_and_pos.append(
                [
                    _quantize_note(note.start, unit) - note_shift,
                    _quantize_note(note.end, unit) - note_shift,
                    _pitch_to_number(note.pitch, meta),
                    note.velocity,
                ]
            )
    if not all_notes_and_pos:
        raise ValueError("No quantizable melody notes after preprocessing.")

    return _construct_melo_sequence(all_notes_and_pos), tempo


def _estimate_key_from_midi_path(midi_path: str) -> tuple[str | None, str | None]:
    midi = PrettyMIDI(midi_path)
    if not midi.instruments or not midi.instruments[0].notes:
        return None, None
    pc = np.zeros(12, dtype=np.float64)
    for n in midi.instruments[0].notes:
        if n.start is not None and n.end is not None:
            w = max(float(n.end - n.start), 1e-4)
            pc[n.pitch % 12] += w
    s = float(pc.sum())
    if s < 1e-6:
        return None, None
    ch = pc / s

    def best_for(profile: np.ndarray) -> tuple[int, float]:
        p = profile / (np.linalg.norm(profile) + 1e-9)
        best_i, best_c = 0, -1.0
        for shift in range(12):
            rolled = np.roll(p, shift)
            c = float(np.dot(ch, rolled))
            if c > best_c:
                best_c, best_i = c, shift
        return best_i, best_c

    maj_i, maj_c = best_for(_KK_MAJOR)
    min_i, min_c = best_for(_KK_MINOR)
    if maj_c >= min_c:
        return _TONIC_SHARP[maj_i], "maj"
    return _TONIC_SHARP[min_i], "min"


def _normalize_bar_count(total_bars: int) -> int:
    n = max(4, int(total_bars))
    r = n % 4
    if r:
        n += 4 - r
    return n


def _partition_bars(total_bars: int) -> list[int]:
    """Split bar count into allowed phrase lengths; prefer larger chunks (fewer phrases)."""
    total_bars = _normalize_bar_count(total_bars)
    allowed = tuple(sorted(ALLOWED_PHRASE_BARS, reverse=True))
    parts: list[int] = []
    rem = total_bars
    while rem > 0:
        chosen = None
        for a in allowed:
            if a > rem:
                continue
            tail = rem - a
            if tail == 0 or tail >= 4:
                chosen = a
                break
        if chosen is None:
            chosen = 4
        parts.append(chosen)
        rem -= chosen
    for p in parts:
        if p not in ALLOWED_PHRASE_BARS:
            raise RuntimeError(f"Invalid phrase partition {parts!r}")
    return parts


def _pick_downbeat_grid(
    beat_times: np.ndarray,
    sec_per_bar: float,
    key_hints: dict | None = None,
) -> np.ndarray:
    """Pick downbeat times: madmom beat_numbers, librosa onset phase hint, else grid search."""
    bt = np.asarray(beat_times, dtype=np.float64).ravel()
    if bt.size == 0:
        return bt
    kh = key_hints or {}
    nums = kh.get("beat_numbers")
    if isinstance(nums, (list, tuple)) and len(nums) == len(bt):
        ni = np.array([int(round(float(x))) for x in nums], dtype=np.int64)
        mask = ni == 1
        if np.any(mask):
            return bt[mask]
    ph = kh.get("beat_downbeat_phase")
    if ph is not None and bt.size >= 4:
        try:
            q = int(ph) % 4
            return bt[q::4]
        except (TypeError, ValueError):
            pass
    if bt.size < 8 or sec_per_bar <= 0:
        return bt[::4]
    best_q, best_cost = 0, 1e18
    for q in range(4):
        dbt = bt[q::4]
        if dbt.size < 3:
            continue
        klim = min(28, dbt.size)
        cost = 0.0
        for n in range(1, klim):
            t = n * sec_per_bar
            cost += float(np.min(np.abs(dbt - t)))
        if cost < best_cost:
            best_cost, best_q = cost, q
    return bt[best_q::4]


def _boundary_reward(
    i: int,
    n_bars: int,
    downbeats: np.ndarray,
    sec_per_bar: float,
    tau: float,
) -> float:
    if i <= 0 or i >= n_bars or downbeats.size == 0:
        return 0.0
    t = i * sec_per_bar
    d = float(np.min(np.abs(downbeats - t)))
    return math.exp(-d / max(tau, 1e-4))


def _detect_pickup_beat(
    midi_path: str,
    tempo: float,
    beat_times_sec: list | None = None,
    *,
    beat_downbeat_phase: int | None = None,
    beat_numbers: list | None = None,
) -> int:
    """
    Detect if the song starts with a pickup (anacrusis) by analyzing the first notes
    relative to the beat grid. Returns the number of 16th-note units to shift.
    
    The key insight: the first downbeat (beat 1 of the first complete bar) should align
    with a strong beat in the audio. If MIDI notes start before this, it's a pickup.
    
    Strategy:
    1. Use beat tracking to find the metrical grid
    2. Identify the first likely downbeat (should be strong and aligned)
    3. Check if MIDI notes start before this downbeat
    4. Calculate the pickup duration in 16th notes
    """
    try:
        midi = PrettyMIDI(midi_path)
        if not midi.instruments or not midi.instruments[0].notes:
            return 0
        
        notes = sorted(midi.instruments[0].notes, key=lambda n: n.start)
        if len(notes) == 0:
            return 0
        
        # Calculate time units
        unit = 60.0 / tempo / 4.0  # seconds per 16th note
        beat_duration = unit * 4  # seconds per quarter note
        bar_duration = beat_duration * 4  # seconds per bar (4/4 time)
        
        first_note_time = notes[0].start
        
        # Method 1: Use beat tracking data if available
        if beat_times_sec and len(beat_times_sec) >= 8:
            beat_times = np.asarray(beat_times_sec, dtype=np.float64)

            first_downbeat: float | None = None
            if (
                beat_numbers is not None
                and isinstance(beat_numbers, (list, tuple))
                and len(beat_numbers) == len(beat_times)
            ):
                for i, bn in enumerate(beat_numbers):
                    if int(round(float(bn))) == 1:
                        first_downbeat = float(beat_times[i])
                        break
            if first_downbeat is None and beat_downbeat_phase is not None:
                try:
                    q = int(beat_downbeat_phase) % 4
                    dbt = beat_times[q::4]
                    if len(dbt) > 0:
                        first_downbeat = float(dbt[0])
                except (TypeError, ValueError):
                    first_downbeat = None

            if first_downbeat is None:
                best_phase = 0
                best_score = -1
                for phase in range(min(4, len(beat_times))):
                    if phase >= len(beat_times):
                        break

                    downbeat_candidates = beat_times[phase::4]

                    if len(downbeat_candidates) < 2:
                        continue

                    if len(downbeat_candidates) >= 3:
                        bar_durs = np.diff(downbeat_candidates)
                        expected_bar_dur = 240.0 / tempo  # 4 beats * 60/tempo
                        regularity = 1.0 / (1.0 + np.std(bar_durs - expected_bar_dur))

                        time_penalty = 1.0 if downbeat_candidates[0] < bar_duration * 1.5 else 0.5

                        score = regularity * time_penalty

                        if score > best_score:
                            best_score = score
                            best_phase = phase

                downbeats = beat_times[best_phase::4]
                if len(downbeats) > 0:
                    first_downbeat = float(downbeats[0])

            if first_downbeat is not None:
                tolerance = 0.1  # 100ms tolerance

                if first_note_time < first_downbeat - tolerance:
                    time_before = first_downbeat - first_note_time
                    pickup_16ths = int(round(time_before / unit))

                    if 0 < pickup_16ths <= 15:
                        logging.info(
                            f"Beat tracking: Detected pickup of {pickup_16ths} sixteenth notes "
                            f"(first note at {first_note_time:.3f}s, first downbeat at {first_downbeat:.3f}s)"
                        )
                        return pickup_16ths

        # Method 2: Analyze MIDI note patterns directly
        # Look at the first bar's note density and position
        first_bar_notes = [n for n in notes if n.start < bar_duration * 1.5]
        
        if len(first_bar_notes) >= 1:
            # Quantize first note to 16th note grid
            first_note_16th = int(round(first_note_time / unit))
            
            # If the first note starts in the second half of a theoretical bar (position 8-15),
            # it's likely a pickup
            if first_note_16th >= 8 and first_note_16th <= 15:
                # Calculate how much to shift to align to next bar
                pickup = 16 - first_note_16th
                
                # Additional validation: check if subsequent notes support this interpretation
                # Count notes in what would be the pickup vs. first complete bar
                pickup_end = first_note_time + (pickup * unit)
                pickup_notes = [n for n in notes if n.start < pickup_end]
                first_bar_notes = [n for n in notes if pickup_end <= n.start < pickup_end + bar_duration]
                
                # The first complete bar should typically have more or equal notes than the pickup
                if len(first_bar_notes) >= len(pickup_notes) * 0.5:  # at least half as dense
                    logging.info(f"MIDI pattern: Detected pickup of {pickup} sixteenth notes "
                               f"(first note at position {first_note_16th}, pickup has {len(pickup_notes)} notes, "
                               f"first bar has {len(first_bar_notes)} notes)")
                    return pickup
        
        # No pickup detected
        logging.info("No pickup detected - song appears to start on the downbeat")
        return 0
        
    except Exception as e:
        logging.warning(f"Pickup detection failed: {e}")
        return 0


def apply_pickup_shift_to_midi(midi_bytes: bytes, shift_16ths: int, tempo: float) -> bytes:
    """
    Apply pickup shift to MIDI bytes by prepending silence.
    
    Args:
        midi_bytes: Original MIDI data
        shift_16ths: Number of 16th notes to shift (prepend silence before first note)
        tempo: Tempo in BPM
    
    Returns:
        Modified MIDI bytes with prepended silence
    """
    if shift_16ths <= 0:
        return midi_bytes
    
    try:
        # Load the original MIDI
        midi = PrettyMIDI(io.BytesIO(midi_bytes))
        
        # Calculate time shift in seconds
        unit = 60.0 / tempo / 4.0  # seconds per 16th note
        time_shift = shift_16ths * unit
        
        # Shift all notes forward in time
        for instrument in midi.instruments:
            for note in instrument.notes:
                if note.start is not None:
                    note.start += time_shift
                if note.end is not None:
                    note.end += time_shift
        
        # Write to bytes
        output = io.BytesIO()
        midi.write(output)
        output.seek(0)
        return output.read()
        
    except Exception as e:
        logging.warning(f"Failed to apply pickup shift to MIDI: {e}")
        return midi_bytes


def _infer_meter_from_beats(beat_times: np.ndarray) -> str | None:
    """Rough meter guess from beat grid regularity (returns '3/4' or '4/4')."""
    bt = np.asarray(beat_times, dtype=np.float64).ravel()
    if bt.size < 12:
        return None

    def score(group: int) -> float:
        best = 1e18
        for phase in range(group):
            db = bt[phase::group]
            if db.size < 4:
                continue
            spac = np.diff(db)
            med = float(np.median(spac)) if spac.size else 0.0
            if med <= 0:
                continue
            # lower is better: consistency of bar durations + fit to linear bar grid
            var = float(np.mean(np.abs(spac - med)))
            lin = np.arange(db.size, dtype=np.float64) * med + db[0]
            fit = float(np.mean(np.abs(db - lin)))
            sc = var + 0.7 * fit
            if sc < best:
                best = sc
        return best

    s3 = score(3)
    s4 = score(4)
    if not np.isfinite(s3) and not np.isfinite(s4):
        return None
    if not np.isfinite(s3):
        return "4/4"
    if not np.isfinite(s4):
        return "3/4"
    # require margin to avoid noisy flips
    if s3 < 0.92 * s4:
        return "3/4"
    return "4/4"
def _partition_bars_beat(
    total_bars: int,
    tempo_midi: float,
    key_hints: dict,
) -> tuple[list[int], bool]:
    """
    Prefer phrase boundaries that line up with instrumental downbeats.
    Returns (lengths, used_beat_track).
    """
    raw = key_hints.get("beat_times_sec")
    bpm_bt = key_hints.get("beat_bpm")
    if not raw or bpm_bt is None:
        return _partition_bars(total_bars), False
    try:
        bpm_bt = float(bpm_bt)
    except (TypeError, ValueError):
        return _partition_bars(total_bars), False
    if not (56.0 < bpm_bt < 200.0):
        return _partition_bars(total_bars), False

    beat_times = np.asarray(raw, dtype=np.float64).ravel()
    if beat_times.size < 8:
        return _partition_bars(total_bars), False

    n = _normalize_bar_count(total_bars)
    tempo_blend = max(48.0, min(190.0, 0.42 * float(tempo_midi) + 0.58 * bpm_bt))
    sec_per_bar = 240.0 / tempo_blend
    tau = max(0.07 * sec_per_bar, 0.05)

    downbeats = _pick_downbeat_grid(beat_times, sec_per_bar, key_hints)

    neg = -1e30
    dp = [neg] * (n + 1)
    parent: list[int | None] = [None] * (n + 1)
    dp[0] = 0.0
    allowed = ALLOWED_PHRASE_BARS

    for i in range(1, n + 1):
        for L in allowed:
            if i < L:
                continue
            prev = i - L
            if dp[prev] <= neg / 2:
                continue
            rw = _boundary_reward(i, n, downbeats, sec_per_bar, tau)
            tie = 1e-12 * float(L)
            sc = dp[prev] + rw + tie
            if sc > dp[i]:
                dp[i] = sc
                parent[i] = L

    if dp[n] <= neg / 2 or parent[n] is None:
        return _partition_bars(total_bars), False

    parts: list[int] = []
    pos = n
    while pos > 0:
        L = parent[pos]
        if L is None or L <= 0:
            return _partition_bars(total_bars), False
        parts.append(L)
        pos -= L
    parts.reverse()
    if sum(parts) != n:
        return _partition_bars(total_bars), False
    for p in parts:
        if p not in ALLOWED_PHRASE_BARS:
            return _partition_bars(total_bars), False
    return parts, True


def _phrases_ui(lengths: list[int]) -> list[dict[str, Any]]:
    out = []
    for i, ln in enumerate(lengths):
        out.append({"phrase_name": _PHRASE_LABELS[i % len(_PHRASE_LABELS)], "phrase_length": ln})
    return out


def analyze_melody_bytes(midi_bytes: bytes, key_hints: dict | None = None) -> dict[str, Any]:
    """
    key_hints: optional tonic/mode from instrumental stem; optional beat_bpm / beat_times_sec
    from no_vocals beat tracking for tempo blend and phrase-boundary nudging.
    """
    key_hints = key_hints or {}
    path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tf:
            tf.write(midi_bytes)
            path = tf.name

        st_hint = key_hints.get("suggested_tonic")
        sm_hint = key_hints.get("suggested_mode")
        tonic_guess, mode_guess = st_hint, sm_hint
        if not tonic_guess or not mode_guess:
            t2, m2 = _estimate_key_from_midi_path(path)
            tonic_guess = tonic_guess or t2 or "C"
            mode_guess = mode_guess or m2 or "maj"

        meta = {"tonic": tonic_guess, "mode": mode_guess, "meter": "4/4"}
        
        # First pass: get tempo without shift
        melo_sequence_temp, tempo = _full_melo_sequence(path, meta, note_shift=0)
        
        # Detect pickup beat (anacrusis) before first downbeat
        pickup_shift = _detect_pickup_beat(
            path,
            tempo,
            key_hints.get("beat_times_sec"),
            beat_downbeat_phase=key_hints.get("beat_downbeat_phase"),
            beat_numbers=key_hints.get("beat_numbers"),
        )
        
        # Second pass: apply the detected shift
        if pickup_shift > 0:
            melo_sequence, _ = _full_melo_sequence(path, meta, note_shift=pickup_shift)
            logging.info(f"Applied pickup shift of {pickup_shift} sixteenth notes")
        else:
            melo_sequence = melo_sequence_temp
        
        total_bars = len(melo_sequence) // 16
        lengths, beat_partition = _partition_bars_beat(total_bars, tempo, key_hints)
        tempo_out = float(tempo)
        bb = key_hints.get("beat_bpm")
        if bb is not None:
            try:
                bbf = float(bb)
                if 56.0 < bbf < 200.0:
                    tempo_out = 0.42 * float(tempo) + 0.58 * bbf
            except (TypeError, ValueError):
                pass
        meter_guess = _infer_meter_from_beats(np.asarray(key_hints.get("beat_times_sec", []), dtype=np.float64))
        out: dict[str, Any] = {
            "auto_phrases": _phrases_ui(lengths),
            "detected_tempo": tempo_out,
            "meter": "4/4",
            "meter_forced": True,
            "meter_guess": meter_guess or "4/4",
            "beat_tracked": beat_partition,
            "pickup_shift": pickup_shift,
        }
        if key_hints.get("beat_bpm") is not None:
            try:
                out["beat_bpm"] = float(key_hints["beat_bpm"])
            except (TypeError, ValueError):
                pass
        if st_hint and sm_hint:
            out["suggested_tonic"] = st_hint
            out["suggested_mode"] = sm_hint
        else:
            out["suggested_tonic"] = tonic_guess
            out["suggested_mode"] = mode_guess
        
        return out
    except Exception:
        logging.exception("melody_analyze failed")
        return {
            "auto_phrases": [{"phrase_name": "A", "phrase_length": 8}],
            "detected_tempo": 120.0,
            "meter": "4/4",
            "meter_forced": True,
            "meter_guess": "4/4",
            "suggested_tonic": key_hints.get("suggested_tonic") or "C",
            "suggested_mode": key_hints.get("suggested_mode") or "maj",
            "beat_tracked": False,
            "pickup_shift": 0,
        }
    finally:
        if path:
            try:
                os.unlink(path)
            except OSError:
                pass


def build_response_more(hints: dict, analysis: dict) -> list[tuple[str, Any]]:
    rows: list[tuple[str, Any]] = []
    src = hints.get("melody_source")
    if src:
        rows.append(("melody_source", src))
    rsn = hints.get("melody_source_reason")
    if rsn:
        rows.append(("melody_source_reason", rsn))
    t = hints.get("suggested_tonic") or analysis.get("suggested_tonic")
    m = hints.get("suggested_mode") or analysis.get("suggested_mode")
    if t:
        rows.append(("suggested_tonic", t))
    if m:
        rows.append(("suggested_mode", m))
    if analysis.get("auto_phrases"):
        rows.append(("auto_phrases", analysis["auto_phrases"]))
    if analysis.get("detected_tempo") is not None:
        rows.append(("detected_tempo", int(round(float(analysis["detected_tempo"])))))
    if analysis.get("meter"):
        rows.append(("meter", analysis["meter"]))
    if analysis.get("meter_guess"):
        rows.append(("meter_guess", analysis["meter_guess"]))
    if analysis.get("beat_bpm") is not None:
        rows.append(("beat_bpm", int(round(float(analysis["beat_bpm"])))))
    if analysis.get("beat_tracked"):
        rows.append(("beat_tracked", True))
    if analysis.get("pickup_shift") is not None and analysis.get("pickup_shift") > 0:
        rows.append(("pickup_shift", analysis["pickup_shift"]))
        # Add a human-readable description
        beats = analysis["pickup_shift"] / 4.0
        rows.append(("pickup_info", f"Song starts {beats:.2f} beat(s) before first downbeat"))
    return rows
