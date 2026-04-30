"""Scale degree maps and tonic string to pitch-class (0-11) for lead-sheet key analysis."""
from __future__ import annotations

# Tonic name (sharps) -> pitch class
str_to_root: dict[str, int] = {
    "C": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
    "Cb": 11,
}

# Diatonic scale degree (1..7) for major / natural minor, by pitch class offset from tonic
major_map: dict[int, float] = {
    0: 1,
    1: 1.5,
    2: 2,
    3: 2.5,
    4: 3,
    5: 4,
    6: 4.5,
    7: 5,
    8: 5.5,
    9: 6,
    10: 6.5,
    11: 7,
    -1: 0,
}
minor_map: dict[int, float] = {
    0: 1,
    1: 1.5,
    2: 2,
    3: 3,
    4: 3.5,
    5: 4,
    6: 4.5,
    7: 5,
    8: 6,
    9: 6.5,
    10: 7,
    11: 7.5,
    -1: 0,
}
