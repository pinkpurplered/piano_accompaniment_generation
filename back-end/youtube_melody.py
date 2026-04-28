"""Download audio from YouTube, split vocals vs accompaniment, transcribe vocals with Basic Pitch.

Accompaniment (no_vocals) is used for coarse key estimation to suggest tonic/mode in the API,
and optional beat tracking for tempo / phrase cuts / chord downbeat alignment:
``madmom`` DBN downbeat tracking when installed (``pip install Cython`` then
``pip install --no-build-isolation -r requirements-madmom.txt`` from ``back-end/``), otherwise librosa
beats plus onset-strength downbeat phase (librosa alone has no measure position).
Piano accompaniment is generated from the lead MIDI (LLaMA-MIDI) after this step; the
cleaner vocal stem reduces bleed from backing instruments into the estimated lead line.
"""
from __future__ import annotations

import importlib.util
import io
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from urllib.parse import urlparse

import numpy as np

_YOUTUBE_HOST = re.compile(
    r"^(www\.)?(youtube\.com|youtu\.be|m\.youtube\.com|music\.youtube\.com)$",
    re.I,
)


def is_allowed_youtube_url(url: str) -> bool:
    if not url or not isinstance(url, str):
        return False
    u = url.strip()
    if len(u) > 500:
        return False
    try:
        p = urlparse(u)
    except Exception:
        return False
    if p.scheme not in ("http", "https"):
        return False
    host = (p.hostname or "").lower()
    return bool(_YOUTUBE_HOST.match(host))


def _ffmpeg_location_dir() -> str | None:
    """Directory containing ffmpeg/ffprobe for yt-dlp --ffmpeg-location."""
    loc = (os.environ.get("FFMPEG_LOCATION") or "").strip()
    if loc:
        if os.path.isdir(loc):
            return os.path.abspath(loc)
        if os.path.isfile(loc) and "ffmpeg" in os.path.basename(loc).lower():
            return os.path.dirname(os.path.abspath(loc))
    w = shutil.which("ffmpeg")
    if w:
        return os.path.dirname(os.path.abspath(w))
    for cand in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"):
        if os.path.isfile(cand):
            return os.path.dirname(cand)
    return None


def _js_runtime_args() -> list[str]:
    """Configure JavaScript engine for yt-dlp signature extraction on bot-protected videos.
    
    yt_dlp_ejs is required for extracting signatures on strict YouTube videos.
    This function ensures yt-dlp can use JavaScript for signature extraction.
    """
    try:
        import yt_dlp_ejs  # noqa: F401
        # yt_dlp_ejs is available; yt-dlp will auto-detect and use it
        # We can optionally force it via extractor-args, but auto-detection usually works
        logging.debug("yt_dlp_ejs is available; JavaScript signature extraction enabled")
        return []
    except ImportError:
        logging.warning(
            "yt_dlp_ejs not found. JavaScript signature extraction disabled. "
            "Some YouTube videos may fail bot detection. "
            "Install with: pip install yt-dlp-ejs"
        )
        return []


def _cookies_from_browser_args() -> list[str]:
    """Try to extract cookies from available browsers for YouTube auth. Returns args or empty list.
    
    Verifies that a browser's cookie database actually exists AND is accessible before using it.
    On macOS, checks for browser profiles in standard locations.
    Skips Safari as its cookies are in a restricted sandboxed container.
    Falls back to empty list if no accessible browsers found - yt-dlp will use alternative methods.
    """
    home = os.path.expanduser("~")
    
    # Browser cookie database locations on macOS
    # Note: Safari excluded - cookies are in restricted ~/Library/Containers/com.apple.Safari/
    browser_paths = {
        "chrome": os.path.join(home, "Library/Application Support/Google/Chrome/Default/Cookies"),
        "firefox": os.path.join(home, ".mozilla/firefox"),  # Firefox has .default-release or other profiles
        "edge": os.path.join(home, "Library/Application Support/Microsoft Edge/Default/Cookies"),
        "chromium": os.path.join(home, "Library/Application Support/Chromium/Default/Cookies"),
    }
    
    for browser, path in browser_paths.items():
        try:
            # Check both existence and readability
            if os.path.exists(path) and os.access(path, os.R_OK):
                logging.debug(f"Found accessible browser cookies database for: {browser}")
                return ["--cookies-from-browser", browser]
            else:
                reason = "not found" if not os.path.exists(path) else "not readable"
                logging.debug(f"Browser {browser} cookies database {reason} at {path}")
        except (OSError, PermissionError) as e:
            logging.debug(f"Cannot access {browser} cookies at {path}: {e}")
    
    # If no browsers with accessible cookies found, return empty list
    # yt-dlp will attempt alternative YouTube access methods (web extraction, etc)
    logging.debug("No accessible browser cookies found; YouTube download will attempt without cookies")
    return []


def _enriched_path_env() -> dict[str, str]:
    env = os.environ.copy()
    parts = [p for p in env.get("PATH", "").split(os.pathsep) if p]
    for extra in ("/opt/homebrew/bin", "/usr/local/bin"):
        if extra not in parts:
            parts.insert(0, extra)
    env["PATH"] = os.pathsep.join(parts)
    return env


def _run(cmd: list[str], cwd: str | None = None, timeout: int = 600) -> None:
    """Run command and raise on error with helpful bot detection messages."""
    r = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=_enriched_path_env(),
    )
    if r.returncode != 0:
        raw = (r.stderr or r.stdout or "").strip()
        if raw:
            # Demucs prints progress bars to stderr; keep the tail where the real failure usually appears.
            tail = "\n".join(raw.splitlines()[-40:])
            
            # Check for YouTube bot detection errors with multiple patterns
            bot_detection_indicators = [
                "Sign in to confirm you're not a bot",
                "Please sign in to confirm you're not a bot",
                "bot detection",
            ]
            has_bot_detection = any(indicator.lower() in tail.lower() for indicator in bot_detection_indicators)
            
            if has_bot_detection:
                err = (
                    f"exit_code={r.returncode}; YouTube bot detection triggered. "
                    f"This video requires authentication or has regional restrictions. "
                    f"To resolve: (1) Install Chrome/Firefox with YouTube login, or (2) Try a different video. "
                    f"Last output:\n{tail}"
                )
            else:
                err = f"exit_code={r.returncode}; last_output:\n{tail}"
        else:
            err = f"exit_code={r.returncode}; command failed: {' '.join(cmd[:6])} ..."
        raise RuntimeError(err[:2400])


def _ytdlp_base_cmd() -> list[str]:
    """
    Return a yt-dlp command that works even when shell PATH is not fully initialized.
    Prefer current interpreter module execution, then fallback to yt-dlp executable.
    """
    try:
        import yt_dlp  # noqa: F401
        return [sys.executable, "-m", "yt_dlp"]
    except Exception:
        pass

    exe = shutil.which("yt-dlp")
    if exe:
        return [exe]
    for cand in (
        "/opt/anaconda3/envs/accomontage2/bin/yt-dlp",
        "/opt/anaconda3/bin/yt-dlp",
        "/usr/local/bin/yt-dlp",
        "/opt/homebrew/bin/yt-dlp",
    ):
        if os.path.isfile(cand):
            return [cand]
    raise RuntimeError("yt-dlp is not installed in the active runtime. Install with: pip install yt-dlp")


# Krumhansl–Kessler weights (C = index 0), for key estimation from instrumental stem
_KK_MAJOR = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88], dtype=np.float64
)
_KK_MINOR = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17], dtype=np.float64
)
_TONIC_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _beat_track_madmom(wav_path: str, max_seconds: float = 240.0) -> dict | None:
    """Return beat grid with true downbeat labels when madmom is installed."""
    if os.environ.get("ACCOMONTAGE_NO_MADMOM", "").lower() in ("1", "true", "yes"):
        return None
    try:
        from madmom.features.downbeats import DBNDownBeatTrackingProcessor
    except ImportError:
        return None
    try:
        import soundfile as sf
    except ImportError:
        return None

    tmp_path: str | None = None
    out: np.ndarray | None = None
    try:
        sig, sr = sf.read(wav_path, always_2d=False)
        if sig.ndim > 1:
            sig = np.mean(sig.astype(np.float64), axis=1)
        else:
            sig = sig.astype(np.float64)
        nmax = int(float(max_seconds) * float(sr))
        if sig.shape[0] > nmax:
            sig = sig[:nmax]
        if sig.size < int(sr) * 2:
            return None
        fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        sf.write(tmp_path, sig, int(sr))
        proc = DBNDownBeatTrackingProcessor(beats_per_bar=[4])
        out = np.asarray(proc(tmp_path), dtype=np.float64)
    except Exception as e:
        logging.warning("madmom downbeat tracking failed: %s", e)
        return None
    finally:
        if tmp_path and os.path.isfile(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    if out is None or out.size == 0 or (out.ndim == 2 and out.shape[1] < 2):
        return None
    if out.ndim == 1:
        return None
    times = out[:, 0].astype(np.float64).ravel()
    nums = np.rint(out[:, 1]).astype(np.int64).ravel()
    if times.size < 8:
        return None
    spq = float(np.median(np.diff(times)))
    if spq <= 0:
        return None
    bpm = 60.0 / spq
    if not (56.0 < bpm < 200.0):
        return None
    if times.size > 2000:
        times = times[:2000]
        nums = nums[:2000]
    return {
        "beat_bpm": float(bpm),
        "beat_times_sec": times.tolist(),
        "beat_numbers": [int(x) for x in nums.tolist()],
        "beat_tracker": "madmom",
    }


def _downbeat_phase_from_onsets(bt: np.ndarray, y: np.ndarray, sr: int) -> int | None:
    """
    Librosa emits quarter-note pulses with arbitrary phase vs bar line. Pick phase q in 0..3
    so bt[q::4] aligns with stronger onset energy (typical of kick / downbeats in pop).
    """
    try:
        import librosa
    except ImportError:
        return None
    hop = 512
    try:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    except Exception:
        return None
    if onset_env.size < 2:
        return None
    try:
        beat_frames = librosa.time_to_frames(bt.astype(np.float64), sr=sr, hop_length=hop)
    except Exception:
        return None
    beat_frames = np.clip(beat_frames, 0, onset_env.size - 1)
    scores: list[float] = []
    best_q, best_s = 0, -1.0
    for q in range(4):
        idxs = beat_frames[q::4][:40]
        s = float(np.sum(onset_env[idxs])) if idxs.size else 0.0
        scores.append(s)
        if s > best_s:
            best_s, best_q = s, q
    top = float(np.max(scores))
    if top <= 1e-9:
        return None
    if (float(np.max(scores)) - float(np.min(scores))) < 0.06 * top:
        return None
    return int(best_q)


def _beat_track_no_vocals(wav_path: str, max_seconds: float = 240.0) -> dict | None:
    """
    Beat-track the instrumental stem for tempo / phrase nudging in melody_analyze.
    Disabled when ACCOMONTAGE_NO_BEAT_TRACK is 1/true/yes.
    """
    if os.environ.get("ACCOMONTAGE_NO_BEAT_TRACK", "").lower() in ("1", "true", "yes"):
        return None
    mm = _beat_track_madmom(wav_path, max_seconds=max_seconds)
    if mm is not None:
        logging.info("beat_track: using madmom (downbeat-labeled grid)")
        return mm

    try:
        import librosa
    except ImportError:
        return None
    try:
        y, sr = librosa.load(wav_path, mono=True, sr=22050, duration=max_seconds)
    except Exception as e:
        logging.warning("beat_track load failed: %s", e)
        return None
    if y.size < sr * 2:
        return None
    try:
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    except Exception as e:
        logging.warning("beat_track beat_track failed: %s", e)
        return None
    tarr = np.asarray(tempo, dtype=np.float64).ravel()
    bpm = float(tarr[0]) if tarr.size else 120.0
    if not (56.0 < bpm < 200.0):
        return None
    beat_times = librosa.frames_to_time(beats, sr=sr)
    bt = np.asarray(beat_times, dtype=np.float64).ravel()
    if bt.size < 8:
        return None
    if bt.size > 2000:
        bt = bt[:2000]
    out: dict = {"beat_bpm": bpm, "beat_times_sec": bt.tolist(), "beat_tracker": "librosa"}
    phase = _downbeat_phase_from_onsets(bt, y, sr)
    if phase is not None:
        out["beat_downbeat_phase"] = phase
        logging.info("beat_track: librosa + onset downbeat phase=%s", phase)
    return out


def _estimate_key_from_wav(wav_path: str, max_seconds: float = 120.0) -> tuple[str | None, str | None]:
    """Return (tonic, mode) with mode in {'maj','min'} using chroma + KK profiles, or (None, None)."""
    try:
        import librosa
    except ImportError:
        logging.warning("librosa not installed; cannot detect key from instrumental stem")
        return None, None

    try:
        y, sr = librosa.load(wav_path, mono=True, sr=22050, duration=max_seconds)
    except Exception as e:
        logging.warning(f"Failed to load instrumental stem for key detection: {e}")
        return None, None
    if y.size < sr * 2:
        logging.warning("Instrumental stem too short for key detection")
        return None, None

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    ch = np.maximum(chroma.mean(axis=1), 0.0)
    s = float(ch.sum())
    if s < 1e-6:
        logging.warning("Instrumental stem has no energy; cannot detect key")
        return None, None
    ch = ch / s

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
        tonic, mode = _TONIC_SHARP[maj_i], "maj"
        logging.info(f"✓ Key detected from instrumental stem: {tonic} {mode} (confidence={maj_c:.3f})")
        return tonic, mode
    tonic, mode = _TONIC_SHARP[min_i], "min"
    logging.info(f"✓ Key detected from instrumental stem: {tonic} {mode} (confidence={min_c:.3f})")
    return tonic, mode


def _find_demucs_vocals_pair(sep_root: str) -> tuple[str, str]:
    for root, _dirs, files in os.walk(sep_root):
        if "vocals.wav" in files and "no_vocals.wav" in files:
            v = os.path.join(root, "vocals.wav")
            nv = os.path.join(root, "no_vocals.wav")
            return v, nv
    raise RuntimeError(
        "Demucs finished but vocals.wav / no_vocals.wav were not found under "
        f"{sep_root!r}. Check Demucs version and --two-stems=vocals support."
    )


def _trim_wav_for_basic_pitch(src: str, work_dir: str, max_seconds: int) -> str:
    """Optionally shorten audio before Basic Pitch (env ACCOMONTAGE_BP_MAX_SECONDS, e.g. 240)."""
    if max_seconds <= 0:
        return src
    out = os.path.join(work_dir, "vocal_for_bp_trimmed.wav")
    ff = shutil.which("ffmpeg") or "ffmpeg"
    cmd = [
        ff,
        "-y",
        "-i",
        src,
        "-t",
        str(max_seconds),
        "-acodec",
        "pcm_s16le",
        "-ar",
        "44100",
        out,
    ]
    try:
        _run(cmd, timeout=min(600, max_seconds + 120))
    except Exception as e:
        logging.warning("ACCOMONTAGE_BP_MAX_SECONDS trim failed, using full file: %s", e)
        return src
    return out if os.path.isfile(out) else src


def _run_demucs_vocals_stem(wav_path: str, out_root: str, timeout: int = 1800) -> tuple[str, str]:
    """Run Demucs htdemucs two-stem (vocals / no_vocals). Returns paths to each wav."""
    os.makedirs(out_root, exist_ok=True)
    # Demucs CLI expects an integer segment value in this environment.
    # Keep it safely below the transformer ceiling (~7.8).
    # Use a conservative default to reduce OOM / abrupt aborts on CPU and Apple Silicon.
    segment_raw = os.environ.get("ACCOMONTAGE_DEMUCS_SEGMENT", "4").strip() or "4"
    try:
        demucs_segment = str(max(1, min(7, int(float(segment_raw)))))
    except ValueError:
        demucs_segment = "7"
    cmd = [
        sys.executable,
        "-m",
        "demucs",
        "--two-stems",
        "vocals",
        "-n",
        "htdemucs",
        "--segment",
        demucs_segment,
        "-o",
        out_root,
        wav_path,
    ]
    _run(cmd, timeout=timeout)
    return _find_demucs_vocals_pair(out_root)


def youtube_url_to_midi_bytes(url: str, work_dir: str, use_vocal_only: bool = True) -> tuple[bytes, dict]:
    """
    Download audio with yt-dlp, isolate vocals with Demucs, run Basic Pitch on vocals.
    Returns (midi_bytes, hints). hints may include suggested_tonic and suggested_mode from
    the instrumental stem (Krumhansl-style chroma). When use_vocal_only is True (default),
    Demucs must succeed; there is no fallback to the full mix—callers should handle errors or
    pass use_vocal_only=False to transcribe the downloaded mix instead.
    work_dir: existing directory for temp files (session folder).
    """
    if not is_allowed_youtube_url(url):
        raise ValueError("Only http(s) YouTube / YouTube Music URLs are allowed.")

    os.makedirs(work_dir, exist_ok=True)
    audio_pattern = os.path.join(work_dir, "yt_audio.%(ext)s")

    ff_dir = _ffmpeg_location_dir()
    if not ff_dir:
        raise RuntimeError(
            "ffmpeg/ffprobe not found. Install FFmpeg (e.g. brew install ffmpeg) "
            "or set FFMPEG_LOCATION to the directory containing ffmpeg and ffprobe."
        )

    ytdlp_cmd = _ytdlp_base_cmd() + ["--no-playlist", "--ffmpeg-location", ff_dir]
    ytdlp_cmd.extend(_js_runtime_args())
    
    # Try to use browser cookies if available (Chrome, Firefox, Edge)
    # Safari excluded due to macOS sandboxing restrictions
    cookies_args = _cookies_from_browser_args()
    if cookies_args:
        ytdlp_cmd.extend(cookies_args)
    
    # Comprehensive bot-evasion and fallback strategies for YouTube
    # This uses multiple extraction methods and player clients to bypass bot detection
    ytdlp_cmd.extend(
        [
            # Enable JavaScript runtime for signature extraction (required for strict videos)
            "--js-runtimes", "node",
            
            # Try YouTube web player client with signature extraction
            "--extractor-args",
            "youtube:player_client=web;skip_unavailable_videos=true",
            
            # Use realistic browser user-agent that mimics popular browsers
            "--user-agent",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            
            # Network and retry configuration for robustness against temporary blocks
            "--socket-timeout", "30",
            "--max-sleep-interval", "10",
            "--min-sleep-interval", "2",
            "-R", "10",  # Reasonable retry count
            
            # Skip problematic fragments gracefully instead of failing hard
            "--skip-unavailable-fragments",
            
            # Add headers to look more like a real browser
            "--add-header", "Referer:https://www.youtube.com",
            "--add-header", "Origin:https://www.youtube.com",
            
            # Format selection and audio extraction
            "-f", "bestaudio/best",
            "-x",
            "--audio-format", "wav",
            "--audio-quality", "0",
            "-o", audio_pattern,
            
            url,
        ]
    )

    _run(ytdlp_cmd, timeout=600)

    wav_path = os.path.join(work_dir, "yt_audio.wav")
    if not os.path.isfile(wav_path):
        # yt-dlp may keep original container
        for name in os.listdir(work_dir):
            if name.startswith("yt_audio.") and name.endswith((".wav", ".m4a", ".webm", ".opus", ".mp3")):
                wav_path = os.path.join(work_dir, name)
                break
        else:
            raise RuntimeError("Download finished but no audio file was found.")

    logging.info("youtube_melody: download complete, wav=%s", wav_path)

    force_skip_demucs = os.environ.get("ACCOMONTAGE_YOUTUBE_NO_DEMUCS", "").lower() in (
        "1",
        "true",
        "yes",
    )
    if bool(use_vocal_only) and force_skip_demucs:
        raise RuntimeError(
            "Vocal separation is required (use_vocal_only) but ACCOMONTAGE_YOUTUBE_NO_DEMUCS is set. "
            "Unset ACCOMONTAGE_YOUTUBE_NO_DEMUCS, or pass use_vocal_only=false to transcribe the full mix."
        )

    skip_demucs = force_skip_demucs or (not bool(use_vocal_only))
    hints: dict = {}
    melody_wav: str

    if skip_demucs:
        melody_wav = wav_path
        hints["melody_source"] = "full_mix_user_choice"
    elif importlib.util.find_spec("demucs") is None:
        raise RuntimeError(
            "Vocal separation requires Demucs. Install with: pip install demucs"
        )
    else:
        sep_root = os.path.join(work_dir, "demucs_stems")
        logging.info("youtube_melody: starting Demucs (can take many minutes on CPU)…")
        try:
            vocals_path, no_vocals_path = _run_demucs_vocals_stem(wav_path, sep_root)
        except Exception as e:
            logging.exception("youtube_melody: Demucs vocal separation failed")
            raise RuntimeError(f"Vocal / instrumental separation failed: {e}") from e
        melody_wav = vocals_path
        hints["melody_source"] = "vocal_stem"
        logging.info("youtube_melody: Demucs finished, vocals=%s", melody_wav)
        st, sm = _estimate_key_from_wav(no_vocals_path)
        if st and sm:
            hints["suggested_tonic"] = st
            hints["suggested_mode"] = sm
        beat_info = _beat_track_no_vocals(no_vocals_path)
        if beat_info:
            hints.update(beat_info)

    # If we did not obtain beat hints from no_vocals, try coarse beat tracking on the current source.
    if hints.get("beat_bpm") is None:
        beat_info = _beat_track_no_vocals(melody_wav)
        if beat_info:
            hints.update(beat_info)

    try:
        max_bp = int(os.environ.get("ACCOMONTAGE_BP_MAX_SECONDS", "0") or 0)
    except ValueError:
        max_bp = 0
    if max_bp > 0:
        logging.info(
            "youtube_melody: trimming to first %s s for Basic Pitch (ACCOMONTAGE_BP_MAX_SECONDS)",
            max_bp,
        )
        melody_wav = _trim_wav_for_basic_pitch(melody_wav, work_dir, max_bp)

    from basic_pitch.inference import predict

    midi_tempo = 120.0
    try:
        bb = float(hints.get("beat_bpm")) if hints.get("beat_bpm") is not None else None
        if bb is not None and 30.0 <= bb <= 260.0:
            midi_tempo = bb
    except (TypeError, ValueError):
        pass

    logging.info(
        "youtube_melody: starting Basic Pitch on %s (midi_tempo=%.2f; long CPU step for full songs)",
        melody_wav,
        midi_tempo,
    )
    _, midi, _ = predict(melody_wav, midi_tempo=midi_tempo)
    logging.info("youtube_melody: Basic Pitch finished")

    if midi.instruments and midi.instruments[0].notes:
        notes = midi.instruments[0].notes
        notes.sort(key=lambda x: x.start)
        
        first_start = notes[0].start
        for n in notes:
            if n.start is not None and n.end is not None and n.end - n.start > 0.1:
                first_start = n.start
                break
        
        shift_amount = 0.0
        if hints.get("beat_times_sec") and len(hints["beat_times_sec"]) > 1:
            from scipy.interpolate import interp1d

            raw_bt = [float(x) for x in hints["beat_times_sec"]]
            nums_hint = hints.get("beat_numbers")
            if isinstance(nums_hint, (list, tuple)) and len(nums_hint) == len(raw_bt):
                paired = sorted(zip(raw_bt, nums_hint), key=lambda z: z[0])
                beats = [float(a) for a, _ in paired]
                nums_aligned = [b for _, b in paired]
            else:
                beats = sorted(raw_bt)
                nums_aligned = None

            beat_duration = 60.0 / midi_tempo

            # Log detailed tempo info for debugging
            logging.info(f"📊 Tempo info: detected={midi_tempo:.1f} BPM, beat_duration={beat_duration:.3f}s")
            if beats and len(beats) > 10:
                # Calculate actual inter-beat interval from detected beats
                intervals = [beats[i+1] - beats[i] for i in range(min(10, len(beats)-1))]
                avg_interval = sum(intervals) / len(intervals)
                actual_bpm = 60.0 / avg_interval
                logging.info(f"    Average beat interval (first 10): {avg_interval:.3f}s → {actual_bpm:.1f} BPM")
            
            valid_beats = [i for i, b in enumerate(beats) if b <= first_start + 0.1]
            start_beat_idx = valid_beats[-1] if valid_beats else 0
            
            target_times = [(i - start_beat_idx) * beat_duration for i in range(len(beats))]
            
            f = interp1d(beats, target_times, fill_value="extrapolate")
            
            logging.info("youtube_melody: warping melody time to align with constant tempo %.2f", midi_tempo)
            new_notes = []
            for n in notes:
                n.start = float(f(n.start))
                n.end = float(f(n.end))
                if n.end > 0:
                    n.start = max(0.0, n.start)
                    new_notes.append(n)
            midi.instruments[0].notes = new_notes

            if nums_aligned is not None and len(nums_aligned) == len(beats):
                synced = [(t, nums_aligned[i]) for i, t in enumerate(target_times) if t >= -0.1]
                hints["beat_times_sec"] = [a for a, _ in synced]
                hints["beat_numbers"] = [b for _, b in synced]
            else:
                k0 = next((i for i, t in enumerate(target_times) if t >= -0.1), 0)
                hints["beat_times_sec"] = [t for t in target_times if t >= -0.1]
                if hints.get("beat_downbeat_phase") is not None:
                    try:
                        ph = int(hints["beat_downbeat_phase"]) % 4
                        hints["beat_downbeat_phase"] = (ph - k0) % 4
                    except (TypeError, ValueError):
                        pass
        else:
            shift_amount = first_start
            if shift_amount > 0.5:
                logging.info("youtube_melody: shifting melody by -%.2fs to remove initial silence", shift_amount)
                new_notes = []
                for n in notes:
                    if n.start is not None:
                        n.start -= shift_amount
                    if n.end is not None:
                        n.end -= shift_amount
                    if n.end is not None and n.end > 0:
                        n.start = max(0.0, n.start) if n.start is not None else 0.0
                        new_notes.append(n)
                midi.instruments[0].notes = new_notes
            
            if hints.get("beat_times_sec"):
                hints["beat_times_sec"] = [
                    b - shift_amount for b in hints["beat_times_sec"] if b - shift_amount >= -0.1
                ]

    buf = io.BytesIO()
    midi.write(buf)
    return buf.getvalue(), hints
