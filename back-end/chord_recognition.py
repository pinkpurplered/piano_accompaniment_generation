"""Chroma-based chord recognition from instrumental audio.

Computes a CQT chroma feature, averages it within each half-bar slot, and matches
against a bank of triad / 7th templates. A diatonic bias against the detected
key keeps the output musical, and a 3-tap consensus smoother removes 1-slot
flickers between neighbouring identical chords.
"""
from __future__ import annotations

import logging
import numpy as np


_PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_NOTE_TO_PC = {n: i for i, n in enumerate(_PITCH_NAMES)}

_TEMPLATES: dict[str, np.ndarray] = {
    "maj":  np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], dtype=float),
    "min":  np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0], dtype=float),
    "maj7": np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1], dtype=float),
    "min7": np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0], dtype=float),
    "dom7": np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0], dtype=float),
}

_DIATONIC_MAJOR = {0: "maj", 2: "min", 4: "min", 5: "maj", 7: "maj", 9: "min", 11: "dim"}
_DIATONIC_MINOR = {0: "min", 2: "dim", 3: "maj", 5: "min", 7: "min", 8: "maj", 10: "maj"}


def _build_template_bank(extended: bool = True) -> list[tuple[str, str, np.ndarray]]:
    qualities = ["maj", "min"]
    if extended:
        qualities += ["maj7", "min7", "dom7"]
    bank = []
    for root_pc, root_name in enumerate(_PITCH_NAMES):
        for q in qualities:
            tmpl = np.roll(_TEMPLATES[q], root_pc)
            tmpl = tmpl / (np.linalg.norm(tmpl) + 1e-9)
            bank.append((root_name, q, tmpl))
    return bank


def _diatonic_bias(tonic: str, mode: str, root: str, quality: str) -> float:
    tonic_pc = _NOTE_TO_PC.get(tonic, 0)
    root_pc = _NOTE_TO_PC.get(root, 0)
    interval = (root_pc - tonic_pc) % 12
    diatonic_map = _DIATONIC_MAJOR if mode == "maj" else _DIATONIC_MINOR
    expected = diatonic_map.get(interval)
    if expected is None:
        return 0.85  # chromatic root
    base_q = quality.replace("7", "")
    if base_q == expected or (expected == "maj" and base_q == "dom"):
        return 1.15
    return 0.92


def recognize_chords(
    audio_path: str,
    slot_times: list[float],
    tonic: str = "C",
    mode: str = "maj",
    extended: bool = True,
    smooth: bool = True,
) -> list[tuple[str, str]]:
    """Recognize one chord per slot from instrumental audio.

    Slot i covers [slot_times[i], slot_times[i+1]); the last anchor is the end
    boundary only, so output length is len(slot_times) - 1.
    """
    try:
        import librosa
    except ImportError:
        logging.error("librosa not installed; chord recognition unavailable")
        return []

    if len(slot_times) < 2:
        logging.warning("chord_recognition: need at least 2 slot_times")
        return []

    try:
        y, sr = librosa.load(audio_path, sr=22050, mono=True)
    except Exception as e:
        logging.error("chord_recognition: failed to load %s: %s", audio_path, e)
        return []

    if y.size < sr:
        logging.warning("chord_recognition: audio too short (%.1fs)", y.size / sr)
        return []

    hop = 512
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop, n_chroma=12)
    chroma = chroma / (np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-9)
    frame_times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=hop)

    bank = _build_template_bank(extended=extended)

    chords: list[tuple[str, str]] = []
    chord_scores: list[float] = []
    for i in range(len(slot_times) - 1):
        t0, t1 = slot_times[i], slot_times[i + 1]
        if t1 <= t0:
            chords.append((tonic, mode))
            chord_scores.append(0.0)
            continue
        mask = (frame_times >= t0) & (frame_times < t1)
        if not mask.any():
            chords.append((tonic, mode))
            chord_scores.append(0.0)
            continue
        slot_chroma = chroma[:, mask].mean(axis=1)
        slot_chroma = slot_chroma / (np.linalg.norm(slot_chroma) + 1e-9)
        best_score = -1.0
        best = (tonic, mode)
        for root, quality, tmpl in bank:
            score = float(np.dot(slot_chroma, tmpl)) * _diatonic_bias(tonic, mode, root, quality)
            if score > best_score:
                best_score = score
                best = (root, quality)
        chords.append(best)
        chord_scores.append(best_score)

    if smooth and len(chords) >= 3:
        smoothed = list(chords)
        for i in range(1, len(chords) - 1):
            prev_c, next_c = chords[i - 1], chords[i + 1]
            if prev_c == next_c and prev_c != chords[i]:
                if chord_scores[i] < max(chord_scores[i - 1], chord_scores[i + 1]) * 1.05:
                    smoothed[i] = prev_c
        chords = smoothed

    if chords:
        from collections import Counter
        top = Counter(f"{r}{q}" for r, q in chords).most_common(5)
        logging.info("🎶 Chord recognition: %d slots, top: %s", len(chords), top)

    return chords
