# Accompaniment Engine Architecture

## Overview

The piano accompaniment generation now supports **two engines** that can be switched via the `ACCOMP_ENGINE` environment variable:

1. **`rules` (Default)** — New rule-based arranger combining chord recognition + voice leading
2. **`llama`** — Legacy LLaMA-MIDI text continuation engine

## Using the Rules-Based Engine (Recommended)

The new rules-based engine follows this pipeline:

```
Melody MIDI
    ↓
[Chord Estimation]  ← Viterbi + diatonic templates
    ↓
[Chord Progression]  ← 2-bar segments with smoothing
    ↓
[Piano Arrangement]  ← Texture selection (block/arpeggio/ballad)
    ↓
MIDI Output (block chords + textured arrangement)
```

### Starting the API with Rules Engine

```bash
cd back-end
# Default is rules engine
python app.py

# Or explicitly:
export ACCOMP_ENGINE=rules
python app.py
```

### Advantages of Rules Engine

- **Deterministic**: Same input always produces same chord progression
- **Key-aware**: Respects detected key and mode; no more wrong-key output
- **Beat-aligned**: Chords start on bar boundaries; no timing drift
- **Fast**: No LLM loading or inference; ~1-2 seconds per song
- **Musically coherent**: Voice leading minimizes jarring transitions
- **Configurable**: Can swap texture (block / arpeggio / ballad)
- **Transparent**: Log shows exact chords and arrangement decisions

## Using the LLaMA-MIDI Engine (Legacy)

To switch back to the old LLaMA-MIDI engine:

```bash
export ACCOMP_ENGINE=llama
python app.py
```

### LLaMA-MIDI Configuration

The LLaMA-MIDI engine respects these environment variables:

```bash
export LLAMA_MIDI_MODEL=dx2102/llama-midi  # HuggingFace model ID
export LLAMA_MIDI_MAX_NEW_TOKENS=16000     # Longer = more accompaniment
export LLAMA_MIDI_TEMPERATURE=1.0          # 0.5=deterministic, 1.5=creative
export LLAMA_MIDI_TOP_P=0.92                # Nucleus sampling
export LLAMA_MIDI_DEVICE=cpu                # cpu | cuda | mps
```

### LLaMA-MIDI Performance

- **Slower**: 30-60 seconds per song (LLM inference)
- **Memory-heavy**: ~6-10 GB RAM or GPU
- **Unreliable**: May generate:
  - Wrong-key accompaniment
  - Looping 2-bar patterns
  - Melody duplicates
  - Timing misalignment

## Architecture Details

### Rule-Based Engine (New)

#### 1. Chord Recognition Module (`chord_recognition.py`)

- **Input**: Audio WAV file (accompaniment or full mix)
- **Output**: List of `(time_sec, root, quality)` chords

**Methods:**
- **CREMA** (if installed): Neural network chord recognition
  - `pip install crema`
  - Highly accurate on clean mixes
  - Faster than training-based methods

- **Viterbi Fallback**: HMM-based decoding
  - No external dependencies
  - Uses Krumhansl–Kessler pitch class profiles
  - Constrained to diatonic chords in detected key

#### 2. Melody-to-Chord Module (`melody_to_chord.py`)

- **Input**: Melody MIDI + key hint
- **Output**: Chord progression (bar number, root, quality)

**Algorithm:**
1. Divide melody into 2-bar segments
2. Count pitch classes in each segment (weighted by duration)
3. Score against diatonic chord templates
4. Apply Viterbi smoothing for temporal coherence
5. Snap chords to bar boundaries

#### 3. Piano Arranger Module (`piano_arranger.py`)

- **Input**: Chord progression + beat grid
- **Output**: Note events with timing and velocities

**Textures:**
- **Block**: Sustained whole-note chords (use as lead sheet)
- **Arpeggio**: Flowing Alberti patterns (classic piano style)
- **Ballad**: LH bass on beats 1&3, RH rolled chords on 2&4

**Voice Leading:**
- Minimizes semitone motion between consecutive chords
- Keeps bass note stable
- Adjusts upper voices to closest chord tones

#### 4. Orchestration Module (`accompaniment_generator.py`)

- Routes between engines based on `ACCOMP_ENGINE` env var
- Handles fallbacks and error recovery
- Writes both block and textured MIDI

### Integration Points

**app.py → `begin_llama_midi_thread()`:**
- Now calls `accompaniment_generator.generate_accompaniment()`
- Passes through all detected context (tonic, mode, beat grid)
- Respects `ACCOMP_ENGINE` setting

**Frontend (unchanged):**
- Still shows two download links (chord MIDI, textured MIDI)
- Now they are actually different (block vs arpeggio)

## Configuration & Tuning

### Rules Engine Tuning

Edit these in code or set env vars (if implemented):

- **`bars_per_chord`** in `melody_to_chord.py`: Default 2 bars per chord
  - Increase for slower songs (e.g., 4 bars)
  - Decrease for fast songs (e.g., 1 bar)

- **Texture selection** in `piano_arranger.py`:
  - Auto-selects based on melody density
  - Can force via `default_texture=Texture.BALLAD` etc.

- **Voice leading cost** in `ChordVoicer`:
  - Adjust `_voice_lead()` to prefer certain voice motions

### Debug Output

All engines log extensively to stdout:

```
🎹 Generate accompaniment context: G major, tempo=120.0 BPM, pickup_shift=0
...
✓ Harmonised melody: 8 chords estimated from 16 bars
🎹 Arranging with rules engine: G major @ 120.0 BPM
Arranging 8 chords...
Wrote chord_gen.mid (32 notes)
Wrote textured_chord_gen.mid (156 notes)
✓ Arranged accompaniment: 32 block notes, 156 textured notes
```

## Migration Guide

If you were using the LLaMA-MIDI engine and want to switch:

1. **Update environment:**
   ```bash
   export ACCOMP_ENGINE=rules
   # Remove old LLAMA_MIDI_* vars if present
   ```

2. **No code changes needed** — the app auto-routes

3. **Differences you'll notice:**
   - Faster (1-2 sec vs 30+ sec)
   - Lower memory usage
   - More consistent results
   - Fewer errors/retries in logs

## Troubleshooting

### Rules engine produces wrong chords

- **Cause**: Melody MIDI from Basic Pitch is inaccurate
- **Fix**: Set more conservative key (try `ACCOMP_ENGINE=llama` if melody is very noisy)

### Block MIDI is too sparse

- **Cause**: Chord events are per 2-bar segment
- **Fix**: Decrease `bars_per_chord` to 1 in `melody_to_chord.py`

### Textured MIDI sounds repetitive

- **Cause**: Texture pattern is too simple for song length
- **Fix**: Try `default_texture=Texture.BALLAD` in `piano_arranger.py`

### LLaMA-MIDI is slow

- **Cause**: CPU inference on macOS (safe default to avoid crashes)
- **Fix**: Try `export LLAMA_MIDI_DEVICE=mps` if you have Apple Silicon
  - Note: May crash on very long generations

## Future Improvements

- [ ] CREMA integration (optional, detected in `chord_recognition.py`)
- [ ] Real accompaniment stem chord detection (instead of melody-based)
- [ ] Texture training classifier (learns style from corpus)
- [ ] Pickup measure intro generation
- [ ] Capo and transposition support
- [ ] Humanisation (timing/velocity swing)
- [ ] Export chord chart as JSON
