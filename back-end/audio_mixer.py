"""Mix MIDI accompaniment with vocal audio and export as MP3."""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import numpy as np
import soundfile as sf


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_PREFERRED_SOUNDFONTS = [
    os.path.join(_REPO_ROOT, 'back-end', 'assets', 'soundfonts', 'MuseScore_General.sf3'),
    os.path.join(_REPO_ROOT, 'back-end', 'soundfonts', 'MuseScore_General.sf3'),
    '/usr/share/sounds/sf2/FluidR3_GM.sf2',
    '/usr/share/soundfonts/FluidR3_GM.sf2',
    '/usr/local/share/soundfonts/FluidR3_GM.sf2',
    '/opt/homebrew/share/soundfonts/default.sf2',
]


def _find_ffmpeg() -> str:
    """Locate ffmpeg executable."""
    ff = shutil.which("ffmpeg")
    if ff:
        return ff
    for cand in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"):
        if os.path.isfile(cand):
            return cand
    raise RuntimeError("ffmpeg not found. Install with: brew install ffmpeg")


def _find_executable(name: str, extra_paths: tuple[str, ...] = ()) -> str | None:
    resolved = shutil.which(name)
    if resolved:
        return resolved
    for candidate in extra_paths:
        if os.path.isfile(candidate):
            return candidate
    return None


def list_available_soundfonts() -> list[str]:
    found = []
    for candidate in _PREFERRED_SOUNDFONTS:
        if os.path.isfile(candidate) and candidate not in found:
            found.append(candidate)
    return found


def resolve_soundfont(soundfont: str | None = None) -> str | None:
    explicit = soundfont or os.environ.get('SOUNDFONT_PATH', '').strip()
    if explicit:
        if os.path.isfile(explicit):
            return explicit
        raise RuntimeError(f'SOUNDFONT_PATH does not exist: {explicit}')
    available = list_available_soundfonts()
    return available[0] if available else None


def calculate_audio_db(wav_path: str) -> dict:
    """
    Calculate average dB (decibels) of an audio file.
    
    Args:
        wav_path: Path to WAV file
        
    Returns:
        dict with keys:
            - rms_db: RMS level in dB (relative to full scale, dBFS)
            - peak_db: Peak level in dB (dBFS)
            - mean_db: Mean absolute level in dB (dBFS)
    """
    if not os.path.isfile(wav_path):
        raise FileNotFoundError(f"Audio file not found: {wav_path}")
    
    try:
        audio, sr = sf.read(wav_path)
        
        # Convert to mono if stereo
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        
        # Calculate RMS (Root Mean Square)
        rms = np.sqrt(np.mean(audio ** 2))
        
        # Calculate peak amplitude
        peak = np.max(np.abs(audio))
        
        # Calculate mean absolute amplitude
        mean_abs = np.mean(np.abs(audio))
        
        # Convert to dB (dBFS - decibels relative to full scale)
        # Add epsilon to avoid log(0)
        epsilon = 1e-10
        rms_db = 20 * np.log10(rms + epsilon)
        peak_db = 20 * np.log10(peak + epsilon)
        mean_db = 20 * np.log10(mean_abs + epsilon)
        
        return {
            "rms_db": float(rms_db),
            "peak_db": float(peak_db),
            "mean_db": float(mean_db),
            "file": wav_path,
            "duration_sec": len(audio) / sr,
            "sample_rate": sr
        }
    except Exception as e:
        raise RuntimeError(f"Failed to analyze audio file {wav_path}: {e}")


def normalize_audio_to_target_db(audio: np.ndarray, target_rms_db: float = -15.0, allow_peak_above: bool = False) -> np.ndarray:
    """
    Normalize audio to a target RMS level in dBFS.
    
    Args:
        audio: Audio samples as numpy array
        target_rms_db: Target RMS level in dBFS (default: -15.0)
        allow_peak_above: If True, prioritize RMS target even if peaks exceed -1 dBFS.
                         If False (default), limit peaks to -1 dBFS for safety.
        
    Returns:
        Normalized audio array
    """
    epsilon = 1e-10
    
    # Calculate current RMS
    current_rms = np.sqrt(np.mean(audio ** 2))
    
    if current_rms < epsilon:
        logging.warning("Audio RMS is near zero, skipping normalization")
        return audio
    
    # Calculate current RMS and peak in dB
    current_rms_db = 20 * np.log10(current_rms + epsilon)
    current_peak = np.max(np.abs(audio))
    current_peak_db = 20 * np.log10(current_peak + epsilon)
    
    # Calculate gain needed to reach target RMS
    gain_db = target_rms_db - current_rms_db
    gain_linear = 10 ** (gain_db / 20)
    
    # Predict what the peak would be after normalization
    predicted_peak = current_peak * gain_linear
    predicted_peak_db = 20 * np.log10(predicted_peak + epsilon)
    
    # Apply gain
    normalized = audio * gain_linear
    
    # Check if we need to limit peaks
    if not allow_peak_above and predicted_peak > 10 ** (-1.0 / 20):  # -1 dBFS threshold
        # Apply limiter to maintain -1 dBFS peak
        limiter_gain = 10 ** (-1.0 / 20) / predicted_peak
        normalized = normalized * limiter_gain
        actual_rms = np.sqrt(np.mean(normalized ** 2))
        actual_rms_db = 20 * np.log10(actual_rms + epsilon)
        logging.info(f"Normalized with peak limiter: {current_rms_db:.2f} dBFS → {actual_rms_db:.2f} dBFS (peak capped at -1.0 dBFS)")
    else:
        actual_rms_db = target_rms_db
        logging.info(f"Normalized: {current_rms_db:.2f} dBFS → {actual_rms_db:.2f} dBFS (gain: {gain_db:+.2f} dB, peak: {predicted_peak_db:.2f} dBFS)")
    
    return normalized


def _find_demucs_vocal_stem(stems_root: str) -> str | None:
    """Return first vocals.wav found under demucs output tree."""
    if not os.path.isdir(stems_root):
        return None
    for root, _dirs, files in os.walk(stems_root):
        if "vocals.wav" in files:
            return os.path.join(root, "vocals.wav")
    return None


def _try_generate_vocal_stem_from_source(work_dir: str, timeout: int = 2400) -> str | None:
    """
    Retry Demucs separation for MP3 export using conservative settings.
    Returns vocals.wav path when successful, else None.
    """
    # Ensure work_dir is absolute
    work_dir = os.path.abspath(work_dir)
    logging.info("Attempting to find source audio in: %s", work_dir)
    
    # Check if work_dir exists
    if not os.path.isdir(work_dir):
        logging.error("Work directory does not exist: %s", work_dir)
        return None
    
    # List all files to help debug
    try:
        all_files = os.listdir(work_dir)
        logging.info("Files in work_dir: %s", all_files)
    except Exception as e:
        logging.error("Cannot list work_dir: %s", e)
        return None
    
    source_wav = os.path.join(work_dir, "yt_audio.wav")
    if not os.path.isfile(source_wav):
        logging.warning("yt_audio.wav not found, searching for alternatives...")
        for name in os.listdir(work_dir):
            if name.startswith("yt_audio.") and name.endswith((".wav", ".m4a", ".webm", ".opus", ".mp3")):
                source_wav = os.path.join(work_dir, name)
                logging.info("Found alternative source: %s", name)
                break
        else:
            logging.error("No source audio file found in %s (no yt_audio.* files)", work_dir)
            return None

    logging.info("Using source audio: %s (%.2f MB)", source_wav, os.path.getsize(source_wav) / (1024*1024))
    
    # Check if demucs is available
    try:
        import importlib.util
        if importlib.util.find_spec("demucs") is None:
            logging.error("demucs module is not installed in the current Python environment")
            logging.error("Current Python: %s", sys.executable)
            return None
        logging.info("✓ demucs module found in Python environment: %s", sys.executable)
    except Exception as e:
        logging.error("Error checking for demucs: %s", e)
        return None
    
    stems_root = os.path.join(work_dir, "demucs_stems")
    os.makedirs(stems_root, exist_ok=True)

    # Try multiple robust profiles in order with progressively smaller segments
    attempts = [
        {"model": "htdemucs", "segment": "8", "extra": ["--device", "cpu"], "timeout": 3600},
        {"model": "htdemucs", "segment": "4", "extra": ["--device", "cpu"], "timeout": 3600},
        {"model": "mdx_extra", "segment": "2", "extra": ["--device", "cpu"], "timeout": 3600},
    ]
    for attempt in attempts:
        segment_raw = attempt["segment"]
        try:
            safe_segment = str(max(1, min(7, int(float(segment_raw)))))
        except ValueError:
            safe_segment = "7"

        cmd = [
            sys.executable,
            "-m",
            "demucs",
            "--two-stems",
            "vocals",
            "-n",
            attempt["model"],
            "--segment",
            safe_segment,
            "-o",
            stems_root,
            source_wav,
        ]
        cmd.extend(attempt.get("extra", []))
        try:
            logging.info(
                "Retrying Demucs for MP3: source=%s model=%s segment=%s timeout=%ds",
                os.path.basename(source_wav),
                attempt["model"],
                safe_segment,
                attempt.get("timeout", 2400),
            )
            logging.info("Demucs command: %s", " ".join(cmd[:6]) + " ... " + os.path.basename(source_wav))
            logging.info("Full command: %s", " ".join(cmd))
            logging.info("Working directory for subprocess: %s", os.getcwd())
            logging.info("Source file absolute path: %s", os.path.abspath(source_wav))
            logging.info("Output directory absolute path: %s", os.path.abspath(stems_root))
            
            # Use Popen to avoid buffer issues with large outputs
            import io
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Combine stderr into stdout
                text=True,
                bufsize=1,
            )
            
            # Read output line by line to avoid buffer overflow
            output_lines = []
            try:
                for line in process.stdout:
                    output_lines.append(line)
                    # Only keep last 50 lines
                    if len(output_lines) > 50:
                        output_lines.pop(0)
                
                process.wait(timeout=attempt.get("timeout", 2400))
                result_returncode = process.returncode
                
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                raise
            
            logging.info("Demucs process completed with exit code: %d", result_returncode)
            if output_lines:
                logging.info("Last output lines:\n%s", "".join(output_lines[-20:]))
            
            # Check what was created in stems_root
            created_files = []
            for root, dirs, files in os.walk(stems_root):
                for f in files:
                    created_files.append(os.path.join(root, f))
            logging.info("Files created in demucs_stems: %s", created_files if created_files else "NONE")
            
            # Only consider it successful if exit code is 0
            if result_returncode == 0:
                vocal_wav = _find_demucs_vocal_stem(stems_root)
                if vocal_wav and os.path.isfile(vocal_wav):
                    logging.info("✓ Demucs retry succeeded for MP3 vocals=%s", vocal_wav)
                    return vocal_wav
                else:
                    logging.warning("Demucs exit code 0 but vocals.wav not found in %s", stems_root)
                    logging.warning("Expected to find vocals.wav in subdirectories of: %s", stems_root)
            else:
                logging.error("Demucs failed with exit code %d (model=%s)", result_returncode, attempt["model"])
                
        except subprocess.TimeoutExpired as e:
            logging.error(
                "Demucs retry TIMEOUT after %s seconds (model=%s, segment=%s)",
                attempt.get("timeout", 2400),
                attempt["model"],
                safe_segment,
            )
            # Kill any remaining demucs processes
            try:
                subprocess.run(["pkill", "-f", "demucs"], capture_output=True)
            except:
                pass
                
        except Exception as e:
            logging.error("Demucs retry exception (model=%s): %s", attempt["model"], str(e))

    # No fallback - we require real AI-separated vocals from Demucs only
    logging.error(
        "All Demucs separation attempts failed. Real isolated vocals are required for MP3 generation. "
        "Cannot use simple center-channel extraction as it is not true vocal isolation."
    )
    return None


def midi_to_wav(midi_path: str, output_wav: str, soundfont: str = None) -> None:
    """
    Convert MIDI to WAV using fluidsynth or timidity.
    Falls back to pretty_midi synthesis if neither is available.
    """
    selected_soundfont = resolve_soundfont(soundfont)

    # Try fluidsynth first with an explicit soundfont.
    fs_bin = _find_executable("fluidsynth", (
        "/opt/homebrew/bin/fluidsynth",
        "/usr/local/bin/fluidsynth",
    ))
    if fs_bin and selected_soundfont:
        if os.path.isfile(selected_soundfont):
            try:
                subprocess.run(
                    [fs_bin, "-ni", "-F", output_wav, "-r", "44100", selected_soundfont, midi_path],
                    check=True,
                    capture_output=True,
                    timeout=60,
                )
                if os.path.isfile(output_wav):
                    logging.info("MIDI->WAV via fluidsynth soundfont=%s: %s", selected_soundfont, output_wav)
                    return
            except Exception as e:
                logging.warning("fluidsynth failed: %s", e)
    
    # Try timidity
    tim_bin = _find_executable("timidity", (
        "/opt/homebrew/bin/timidity",
        "/usr/local/bin/timidity",
    ))
    if tim_bin:
        try:
            subprocess.run(
                [tim_bin, midi_path, "-Ow", "-o", output_wav],
                check=True,
                capture_output=True,
                timeout=60,
            )
            if os.path.isfile(output_wav):
                logging.info("MIDI->WAV via timidity: %s", output_wav)
                return
        except Exception as e:
            logging.warning("timidity failed: %s", e)
    
    # Final fallback is not a realistic grand piano. Keep it as a last resort only.
    try:
        import pretty_midi
        pm = pretty_midi.PrettyMIDI(midi_path)
        # Use synthesize() method which doesn't require pyfluidsynth
        audio = pm.synthesize(fs=44100)
        # Normalize to prevent clipping
        max_abs = np.max(np.abs(audio))
        if max_abs > 1e-6:
            audio = audio / max_abs
        audio = audio * 0.8
        sf.write(output_wav, audio, 44100)
        logging.warning("MIDI->WAV fell back to pretty_midi synth (non-soundfont): %s", output_wav)
    except Exception as e:
        raise RuntimeError(f"All MIDI synthesis methods failed: {e}")


def render_soundfont_preview(soundfont_path: str, output_mp3: str) -> None:
    import pretty_midi

    preview_midi = pretty_midi.PrettyMIDI(initial_tempo=92)
    piano = pretty_midi.Instrument(program=0, name='Preview Piano')
    notes = [48, 55, 60, 64, 67, 72, 76, 79]
    start = 0.0
    for pitch in notes:
        piano.notes.append(pretty_midi.Note(velocity=95, pitch=pitch, start=start, end=start + 0.9))
        start += 0.55
    preview_midi.instruments.append(piano)

    preview_dir = os.path.dirname(output_mp3)
    os.makedirs(preview_dir, exist_ok=True)
    preview_midi_path = os.path.join(preview_dir, 'soundfont_preview.mid')
    preview_wav_path = os.path.join(preview_dir, 'soundfont_preview.wav')
    preview_midi.write(preview_midi_path)
    midi_to_wav(preview_midi_path, preview_wav_path, soundfont=soundfont_path)

    ffmpeg = _find_ffmpeg()
    subprocess.run(
        [
            ffmpeg,
            '-y',
            '-i',
            preview_wav_path,
            '-codec:a',
            'libmp3lame',
            '-qscale:a',
            '2',
            output_mp3,
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )

    for temp in [preview_midi_path, preview_wav_path]:
        if os.path.isfile(temp):
            os.remove(temp)


def mix_vocal_and_midi(
    vocal_wav: str,
    midi_path: str,
    output_mp3: str,
    midi_gain: float = 0.7,
    vocal_gain: float = 0.9,
) -> None:
    """
    Mix vocal audio with MIDI accompaniment and export as MP3.
    
    Args:
        vocal_wav: Path to vocal WAV file (MUST be Demucs-separated vocals)
        midi_path: Path to MIDI file
        output_mp3: Path for output MP3
        midi_gain: Volume multiplier for MIDI (0.0-1.0)
        vocal_gain: Volume multiplier for vocals (0.0-1.0)
    """
    # Verify we're using real isolated vocals, not full audio
    vocal_basename = os.path.basename(vocal_wav)
    if "demucs_stems" not in vocal_wav or vocal_basename != "vocals.wav":
        raise RuntimeError(
            f"Invalid vocal source! Must be Demucs-separated vocals.wav, got: {vocal_wav} (filename: {vocal_basename}). "
            f"This prevents accidentally using full song audio (yt_audio.wav) or instrumentals (no_vocals.wav)."
        )
    
    logging.info("✓ Using Demucs-separated vocals for mixing: %s", vocal_wav)
    
    work_dir = os.path.dirname(output_mp3)
    os.makedirs(work_dir, exist_ok=True)
    
    # Convert MIDI to WAV
    midi_wav = os.path.join(work_dir, "temp_midi.wav")
    midi_to_wav(midi_path, midi_wav)
    
    if not os.path.isfile(midi_wav):
        raise RuntimeError(f"MIDI synthesis failed: {midi_wav}")
    
    # Load both audio files
    vocal_audio, vocal_sr = sf.read(vocal_wav)
    midi_audio, midi_sr = sf.read(midi_wav)
    
    # Ensure mono for simplicity
    if vocal_audio.ndim > 1:
        vocal_audio = vocal_audio.mean(axis=1)
    if midi_audio.ndim > 1:
        midi_audio = midi_audio.mean(axis=1)
    
    # Resample if needed (both to 44100 Hz)
    target_sr = 44100
    if vocal_sr != target_sr:
        import librosa
        vocal_audio = librosa.resample(vocal_audio, orig_sr=vocal_sr, target_sr=target_sr)
    if midi_sr != target_sr:
        import librosa
        midi_audio = librosa.resample(midi_audio, orig_sr=midi_sr, target_sr=target_sr)
    
    # Match lengths (trim or pad to the longer one)
    max_len = max(len(vocal_audio), len(midi_audio))
    if len(vocal_audio) < max_len:
        vocal_audio = np.pad(vocal_audio, (0, max_len - len(vocal_audio)))
    if len(midi_audio) < max_len:
        midi_audio = np.pad(midi_audio, (0, max_len - len(midi_audio)))
    
    # Normalize both tracks to target RMS level (-15 dBFS) before mixing
    # This ensures consistent loudness between vocals and accompaniment
    logging.info("=" * 60)
    logging.info("Normalizing audio tracks for mixing...")
    logging.info("=" * 60)
    
    logging.info("Vocals:")
    vocal_audio = normalize_audio_to_target_db(vocal_audio, target_rms_db=-15.0)
    
    logging.info("MIDI accompaniment:")
    midi_audio = normalize_audio_to_target_db(midi_audio, target_rms_db=-15.0)
    
    # Apply user-specified gains (now as final balance adjustment, not primary volume control)
    # Since both tracks are normalized to -15 dBFS, the gains work as relative balance
    if vocal_gain != 1.0:
        logging.info(f"Applying vocal gain adjustment: {vocal_gain:.2f}x")
        vocal_audio = vocal_audio * vocal_gain
    if midi_gain != 1.0:
        logging.info(f"Applying MIDI gain adjustment: {midi_gain:.2f}x")
        midi_audio = midi_audio * midi_gain
    
    # Mix the normalized tracks
    mixed = vocal_audio + midi_audio
    
    # Apply soft limiter to prevent clipping in final mix
    max_val = np.max(np.abs(mixed))
    if max_val > 0.99:
        limiter_ratio = 0.99 / max_val
        mixed = mixed * limiter_ratio
        logging.info(f"Applied final mix limiter: {max_val:.3f} → 0.99 (ratio: {limiter_ratio:.3f})")
    
    logging.info("=" * 60)
    
    # Write temporary WAV
    temp_mixed_wav = os.path.join(work_dir, "temp_mixed.wav")
    sf.write(temp_mixed_wav, mixed, target_sr)
    
    # Convert to MP3 using ffmpeg
    ffmpeg = _find_ffmpeg()
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            temp_mixed_wav,
            "-codec:a",
            "libmp3lame",
            "-qscale:a",
            "2",
            output_mp3,
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )
    
    # Cleanup temp files
    for temp in [midi_wav, temp_mixed_wav]:
        if os.path.isfile(temp):
            os.remove(temp)
    
    logging.info("Mixed MP3 created: %s", output_mp3)


def create_mixed_outputs(
    session_id: str,
    work_dir: str,
    output_mp3_dir: str,
) -> dict:
    """
    Create all mixed MP3 outputs after MIDI generation.
    
    Returns dict with keys:
        - vocal_chord_mp3: vocals mixed with chord MIDI (accompaniment only)
        - vocal_textured_mp3: vocals mixed with textured chord MIDI (accompaniment only)
    """
    logging.info("=" * 60)
    logging.info("Starting MP3 generation for session: %s", session_id)
    logging.info("Work directory: %s", work_dir)
    logging.info("=" * 60)
    
    # Require separated vocal stem so the exported mix is always isolated vocals + generated accompaniment.
    stems_root = os.path.join(work_dir, "demucs_stems")
    logging.info("Looking for Demucs vocals in: %s", stems_root)
    
    # Check if stems_root exists
    if os.path.isdir(stems_root):
        logging.info("✓ demucs_stems directory exists")
        try:
            stem_contents = []
            for root, dirs, files in os.walk(stems_root):
                for f in files:
                    stem_contents.append(os.path.join(root, f))
            logging.info("Files in demucs_stems: %s", stem_contents if stem_contents else "EMPTY")
        except Exception as e:
            logging.warning("Cannot list demucs_stems: %s", e)
    else:
        logging.warning("✗ demucs_stems directory does NOT exist")
    
    vocal_wav = _find_demucs_vocal_stem(stems_root)
    
    if vocal_wav and os.path.isfile(vocal_wav):
        logging.info("✓ Found Demucs-separated vocals for MP3 mix: %s", vocal_wav)
    else:
        logging.warning("Demucs vocals not found in %s, attempting to generate...", stems_root)
        vocal_wav = _try_generate_vocal_stem_from_source(work_dir)
        if vocal_wav and os.path.isfile(vocal_wav):
            logging.info("✓ Successfully generated Demucs-separated vocals: %s", vocal_wav)

    if not vocal_wav or not os.path.isfile(vocal_wav):
        raise RuntimeError(
            f"Real AI-separated vocal stem (Demucs) is required for MP3 generation but was not found in {work_dir}. "
            f"Expected demucs_stems/**/vocals.wav. Simple center-channel extraction is not acceptable. "
            "MP3 export is restricted to isolated vocals + generated grand piano only."
        )
    
    # Additional safety check: ensure path contains demucs_stems and vocals.wav
    vocal_filename = os.path.basename(vocal_wav)
    if "demucs_stems" not in vocal_wav or vocal_filename != "vocals.wav":
        raise RuntimeError(
            f"Invalid vocal source detected: {vocal_wav}. "
            f"Must be from Demucs separation with exact filename 'vocals.wav' in 'demucs_stems' directory. "
            f"Got filename: '{vocal_filename}'. This prevents using the original full song audio (e.g., yt_audio.wav) "
            f"or instrumental track (no_vocals.wav)."
        )
    
    logging.info("✓ Verified Demucs-separated vocals will be used: %s", vocal_wav)
    
    # Locate MIDI files
    chord_midi = os.path.join(work_dir, "chord_gen.mid")
    textured_midi = os.path.join(work_dir, "textured_chord_gen.mid")
    
    if not os.path.isfile(chord_midi):
        raise RuntimeError(f"chord_gen.mid not found: {chord_midi}")
    if not os.path.isfile(textured_midi):
        raise RuntimeError(f"textured_chord_gen.mid not found: {textured_midi}")
    
    # Extract accompaniment-only versions (remove melody track)
    import pretty_midi
    
    chord_midi_acc = os.path.join(work_dir, "chord_gen_acc_only.mid")
    midi_obj = pretty_midi.PrettyMIDI(chord_midi)
    if len(midi_obj.instruments) > 1:
        acc_midi = pretty_midi.PrettyMIDI()
        acc_midi.time_signature_changes = midi_obj.time_signature_changes
        acc_midi.key_signature_changes = midi_obj.key_signature_changes
        for instrument in midi_obj.instruments[1:]:  # Skip first track (melody)
            acc_midi.instruments.append(instrument)
        acc_midi.write(chord_midi_acc)
    else:
        chord_midi_acc = chord_midi  # Use original if only one track
    
    textured_midi_acc = os.path.join(work_dir, "textured_chord_gen_acc_only.mid")
    midi_obj = pretty_midi.PrettyMIDI(textured_midi)
    if len(midi_obj.instruments) > 1:
        acc_midi = pretty_midi.PrettyMIDI()
        acc_midi.time_signature_changes = midi_obj.time_signature_changes
        acc_midi.key_signature_changes = midi_obj.key_signature_changes
        for instrument in midi_obj.instruments[1:]:  # Skip first track (melody)
            acc_midi.instruments.append(instrument)
        acc_midi.write(textured_midi_acc)
    else:
        textured_midi_acc = textured_midi  # Use original if only one track

    def _force_grand_piano_timbre(src_midi: str, dst_midi: str) -> str:
        """Rewrite MIDI instruments to Acoustic Grand Piano (program 0)."""
        midi = pretty_midi.PrettyMIDI(src_midi)
        for instrument in midi.instruments:
            instrument.program = 0
            instrument.is_drum = False
            if not instrument.name:
                instrument.name = "Acoustic Grand Piano"
        midi.write(dst_midi)
        return dst_midi

    chord_midi_piano = _force_grand_piano_timbre(
        chord_midi_acc,
        os.path.join(work_dir, "chord_gen_acc_grand_piano.mid"),
    )
    textured_midi_piano = _force_grand_piano_timbre(
        textured_midi_acc,
        os.path.join(work_dir, "textured_chord_gen_acc_grand_piano.mid"),
    )
    
    os.makedirs(output_mp3_dir, exist_ok=True)
    
    # Generate unique MP3 names
    import time
    timestamp = str(int(time.time() * 1000))
    
    vocal_chord_name = f"{session_id}_{timestamp}_vocal_chord.mp3"
    vocal_textured_name = f"{session_id}_{timestamp}_vocal_textured.mp3"
    
    vocal_chord_path = os.path.join(output_mp3_dir, vocal_chord_name)
    vocal_textured_path = os.path.join(output_mp3_dir, vocal_textured_name)
    
    # Mix and create MP3s (using isolated vocals + grand-piano accompaniment only)
    try:
        mix_vocal_and_midi(vocal_wav, chord_midi_piano, vocal_chord_path)
    except Exception as e:
        logging.error("Failed to create vocal+chord MP3: %s", e)
        raise
    
    try:
        mix_vocal_and_midi(vocal_wav, textured_midi_piano, vocal_textured_path)
    except Exception as e:
        logging.error("Failed to create vocal+textured MP3: %s", e)
        raise
    
    return {
        "vocal_chord_mp3": vocal_chord_name,
        "vocal_textured_mp3": vocal_textured_name,
    }
