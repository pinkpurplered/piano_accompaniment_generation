# Pickup Beat Detection (Anacrusis Detection)

## Overview

The melody analysis system now includes automatic detection of pickup measures (also called anacrusis) - when a song starts on a weak beat before the first downbeat.

## What is a Pickup?

In music, a pickup (or anacrusis) occurs when a piece begins with one or more notes before the first strong beat (downbeat) of the first complete measure. For example:
- A song in 4/4 time might start on beat 4 of an incomplete measure
- The first complete measure then begins after these initial notes
- Common in many popular songs and classical pieces

## How It Works

### Detection Algorithm

The `_detect_pickup_beat()` function uses a two-method approach to determine if there's a pickup:

#### Method 1: Beat Tracking Analysis (Primary, Most Accurate)

When beat tracking data is available from the instrumental stem:

1. **Identify Metrical Grid**
   - Uses beat positions from librosa's beat tracking
   - Tests different phases (0, 1, 2, 3) to find which represents downbeats
   - In 4/4 time, downbeats should occur every 4 beats

2. **Score Each Phase**
   - Checks regularity of bar durations (should be consistent)
   - Prefers phases where first downbeat isn't too early (pickups are typically short)
   - Selects the phase with the best regularity score

3. **Compare First Note to First Downbeat**
   ```python
   first_note_time = 0.5 seconds  (example)
   first_downbeat = 1.8 seconds   (inferred from beat tracking)
   time_before = 1.8 - 0.5 = 1.3 seconds
   pickup_16ths = round(1.3 / (60/tempo/4)) = ~5 sixteenth notes
   ```

4. **Validate Result**
   - Pickup must be 1-15 sixteenth notes (sanity check)
   - Uses 100ms tolerance for timing variations

#### Method 2: MIDI Pattern Analysis (Fallback)

When beat tracking isn't available, analyzes the MIDI directly:

1. **Quantize First Note Position**
   - Converts first note time to 16th-note grid position
   - Example: Note at 0.75s with tempo=120 → position ~6

2. **Detect Incomplete First Bar**
   - If first note is at positions 8-15 (second half of bar), likely a pickup
   - Calculate shift needed: `pickup = 16 - first_note_position`

3. **Validate with Note Density**
   ```
   Example: First note at position 12
   - Pickup measure: positions 12-15 (4 positions, e.g., 3 notes)
   - First complete bar: positions 0-15 (16 positions, e.g., 8 notes)
   - Validation: first bar should have ≥50% note density of pickup
   ```

### Key Improvements Over Original Implementation

1. **Proper Downbeat Inference**: Instead of trying random phases, scores each phase based on:
   - Regularity of bar durations
   - Reasonable first downbeat timing
   
2. **Better Validation**: Checks that:
   - Pickup duration is realistic (1-15 sixteenth notes)
   - Note patterns support the pickup interpretation
   
3. **Dual Methods**: Falls back to MIDI pattern analysis when beat tracking unavailable

### Implementation Details

```python
def _detect_pickup_beat(
    midi_path: str,
    tempo: float,
    beat_times_sec: list | None = None,
) -> int:
    """
    Returns the number of 16th-note units to shift.
    - 0 means no pickup detected (starts on beat 1)
    - 4 means pickup of 1 beat (starts on beat 4 of pickup measure)
    - 8 means pickup of 2 beats
    - etc.
    """
```

### Integration

The pickup detection is integrated into the `analyze_melody_bytes()` function:

1. First pass: Analyze MIDI to get tempo
2. Detect pickup: Determine shift amount using beat tracking if available
3. Second pass: Re-process MIDI with shift applied
4. Result: Notes are properly aligned to downbeats

## Output

The analysis results include:

- `pickup_shift`: Number of 16th notes shifted (0 if no pickup)
- `pickup_info`: Human-readable description (added to API response)

Example output:
```json
{
  "pickup_shift": 4,
  "pickup_info": "Song starts 1.00 beat(s) before first downbeat",
  "detected_tempo": 120,
  "beat_tracked": true,
  ...
}
```

## Logging

The system logs pickup detection for debugging:
```
INFO - Beat tracking: Detected pickup of 4 sixteenth notes (first note at 0.500s, first downbeat at 1.800s)
INFO - Applied pickup shift of 4 sixteenth notes
```

Or if using fallback:
```
INFO - MIDI pattern: Detected pickup of 4 sixteenth notes (first note at position 12, pickup has 3 notes, first bar has 8 notes)
INFO - Applied pickup shift of 4 sixteenth notes
```

## Testing

Use the test script to verify pickup detection:

```bash
cd back-end
/opt/anaconda3/envs/accomontage2/bin/python test_pickup_detection.py [optional_midi_path]
```

## Examples

### Song Starting on Beat 1 (No Pickup)
```
INFO - No pickup detected - song appears to start on the downbeat
Pickup Shift: 0 sixteenth notes
→ Song starts on the first downbeat
```

### Song Starting on Beat 4 (1 Beat Pickup)
```
INFO - Beat tracking: Detected pickup of 4 sixteenth notes (first note at 0.500s, first downbeat at 1.800s)
Pickup Shift: 4 sixteenth notes
→ Song starts 1.00 beat(s) before first downbeat
```

### Song Starting on Beat 3 (2 Beats Pickup)
```
INFO - MIDI pattern: Detected pickup of 8 sixteenth notes (first note at position 8, pickup has 4 notes, first bar has 12 notes)
Pickup Shift: 8 sixteenth notes
→ Song starts 2.00 beat(s) before first downbeat
```

## Benefits

1. **Correct Alignment**: Chord progressions and accompaniment align properly with downbeats
2. **Accurate Phrase Detection**: Phrase boundaries are detected at natural musical positions
3. **Better Generation**: Improved chord generation that respects musical structure
4. **Automatic**: No manual adjustment needed - works automatically for all songs
5. **Robust**: Uses beat tracking when available, falls back to MIDI analysis

## Technical Notes

- Shift is calculated in 16th notes (the base unit for the system)
- Maximum shift is 15 sixteenth notes (just under 1 bar in 4/4)
- Beat tracking method is preferred as it uses actual audio rhythmic information
- MIDI pattern method uses heuristics based on note positions and density
- Both methods include validation to prevent false positives

## Limitations

- Currently optimized for 4/4 time signature
- Beat tracking depends on quality of instrumental stem separation
- Very subtle pickups (1-2 sixteenth notes) may not be detected
- Complex meter changes within the piece are not handled

## Future Improvements

Potential enhancements:
- Support for other time signatures (3/4, 6/8, etc.)
- Integration with madmom or BeatNet for true downbeat detection
- Machine learning-based pickup detection
- User override option in API
- Visualization of detected pickup in UI
- Confidence scores for pickup detection
