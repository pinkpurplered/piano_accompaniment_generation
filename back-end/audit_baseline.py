"""Phase 0: Baseline audit of piano accompaniment rules engine on 5 Cantonese ballads.

Processes YouTube URLs through the full pipeline (upload → generate → download)
and collects metrics: detected key/tempo, chord counts, MIDI/MP3 outputs.
"""

import json
import logging
import os
import shutil
import sys
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import pretty_midi

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

BACKEND_URL = os.environ.get('BACKEND_URL', 'http://127.0.0.1:8765')
API_ROUTE = '/api/chorderator_back_end'
AUDIT_RESULTS_DIR = Path(__file__).parent / 'audit_results'
POLL_INTERVAL_SEC = 2
MAX_POLL_ATTEMPTS = 300  # 10 minutes max


@dataclass
class UrlAuditEntry:
    """Audit results for a single YouTube URL."""
    url: str
    song_name: str
    detected_key: Optional[str] = None
    detected_mode: Optional[str] = None
    detected_tempo: Optional[float] = None
    pickup_shift: Optional[int] = None
    chord_count: int = 0
    block_note_count: int = 0
    textured_note_count: int = 0
    melody_note_count: int = 0
    errors: list = None
    log_output: str = ""
    midi_files: dict = None
    mp3_files: dict = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.midi_files is None:
            self.midi_files = {}
        if self.mp3_files is None:
            self.mp3_files = {}


def http_request(method: str, path: str, data: Optional[str] = None, cookies: Optional[dict] = None, timeout: int = 30) -> tuple[dict, Optional[str]]:
    """Make HTTP request to backend API.

    Returns: (response_dict, new_session_id_if_any)
    """
    url = BACKEND_URL + API_ROUTE + path
    headers = {'Content-Type': 'application/json'}

    if cookies:
        headers['Cookie'] = f"session={cookies.get('session', '')}"

    req = urllib.request.Request(url, method=method)
    for k, v in headers.items():
        req.add_header(k, v)

    if data:
        req.data = data.encode('utf-8')

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            response_data = json.loads(response.read().decode('utf-8'))
            new_session = response.headers.get('Set-Cookie')
            session_id = None
            if new_session and 'session=' in new_session:
                session_id = new_session.split('session=')[1].split(';')[0]
            return response_data, session_id
    except urllib.error.HTTPError as e:
        body = e.read().decode('utf-8')
        try:
            return json.loads(body), None
        except:
            return {'status': 'error', 'error': body}, None
    except Exception as e:
        logger.error(f"HTTP request failed: {e}")
        return {'status': 'error', 'error': str(e)}, None


def process_single_url(url: str) -> UrlAuditEntry:
    """Process a single YouTube URL through upload → generate → download."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing: {url}")
    logger.info(f"{'='*60}")

    entry = UrlAuditEntry(url=url, song_name=url.split('v=')[-1][:11])
    cookies = {'session': ''}

    # Step 1: Upload YouTube URL
    logger.info("Step 1: POST /upload_youtube (this may take 2-3 minutes...)")
    upload_payload = json.dumps({'url': url, 'use_vocal_only': True})
    response, session_id = http_request('POST', '/upload_youtube', data=upload_payload, timeout=300)

    if response.get('status') != 'ok':
        msg = response.get('status', 'unknown error')
        entry.errors.append(f"upload_youtube failed: {msg}")
        logger.error(f"  ✗ Upload failed: {msg}")
        return entry

    if session_id:
        cookies['session'] = session_id
        logger.info(f"  ✓ Session created: {session_id[:8]}...")

    # Extract detected metadata
    entry.detected_key = response.get('detected_tonic')
    entry.detected_mode = response.get('detected_mode', 'maj')
    entry.detected_tempo = response.get('detected_tempo')
    entry.pickup_shift = response.get('pickup_shift', 0)
    logger.info(f"  ✓ Detected: {entry.detected_key} {entry.detected_mode} @ {entry.detected_tempo} BPM")

    # Step 2: Generate accompaniment
    logger.info("Step 2: POST /generate")
    gen_payload = json.dumps({
        'tonic': entry.detected_key or 'C',
        'mode': entry.detected_mode or 'maj',
        'meter': '4/4',
        'tempo': entry.detected_tempo or 120,
    })
    response, _ = http_request('POST', '/generate', data=gen_payload, cookies=cookies, timeout=60)

    if response.get('status') != 'ok':
        msg = response.get('status', 'unknown error')
        entry.errors.append(f"generate failed: {msg}")
        logger.error(f"  ✗ Generate failed: {msg}")
        return entry

    logger.info("  ✓ Generation started (background thread)")

    # Step 3: Poll for completion
    logger.info("Step 3: Poll /stage_query")
    for attempt in range(MAX_POLL_ATTEMPTS):
        response, _ = http_request('GET', '/stage_query', cookies=cookies)
        stage = response.get('stage', '0')

        if response.get('generate_error'):
            entry.errors.append(f"Generate error: {response['generate_error']}")
            logger.error(f"  ✗ Generation error: {response['generate_error']}")
            break

        if stage == '7':
            logger.info(f"  ✓ Generation complete (stage {stage})")
            break

        if attempt % 5 == 0:
            logger.info(f"    Polling... (attempt {attempt}/{MAX_POLL_ATTEMPTS})")
        time.sleep(POLL_INTERVAL_SEC)
    else:
        entry.errors.append(f"Generation timeout (>{MAX_POLL_ATTEMPTS * POLL_INTERVAL_SEC}s)")
        logger.error(f"  ✗ Timeout waiting for generation")
        return entry

    # Step 4: Retrieve generated files
    logger.info("Step 4: GET /generated_query")
    response, _ = http_request('GET', '/generated_query', cookies=cookies)

    if response.get('status') != 'ok':
        msg = response.get('status', 'unknown error')
        entry.errors.append(f"generated_query failed: {msg}")
        logger.error(f"  ✗ Query failed: {msg}")
        return entry

    chord_midi_name = response.get('chord_midi_name')
    acc_midi_name = response.get('acc_midi_name')
    vocal_chord_mp3 = response.get('vocal_chord_mp3')
    vocal_textured_mp3 = response.get('vocal_textured_mp3')

    entry.mp3_files = {
        'vocal_chord': vocal_chord_mp3,
        'vocal_textured': vocal_textured_mp3,
    }

    logger.info(f"  ✓ Block MIDI: {chord_midi_name}")
    logger.info(f"  ✓ Textured MIDI: {acc_midi_name}")

    # Step 5: Download MIDI files and analyze
    logger.info("Step 5: Download and analyze MIDI files")

    # Create song output directory
    song_dir = AUDIT_RESULTS_DIR / f"song_{entry.song_name}"
    song_dir.mkdir(parents=True, exist_ok=True)

    if chord_midi_name:
        midi_path = song_dir / 'chord_gen.mid'
        if download_file(f"{BACKEND_URL}/midi/{chord_midi_name}", str(midi_path)):
            entry.midi_files['chord_gen'] = str(midi_path)
            stats = analyze_midi(str(midi_path))
            entry.block_note_count = stats['note_count']
            logger.info(f"    ✓ Block MIDI: {stats['note_count']} notes, {stats['duration']:.1f}s")

    if acc_midi_name:
        midi_path = song_dir / 'textured_chord_gen.mid'
        if download_file(f"{BACKEND_URL}/midi/{acc_midi_name}", str(midi_path)):
            entry.midi_files['textured_chord_gen'] = str(midi_path)
            stats = analyze_midi(str(midi_path))
            entry.textured_note_count = stats['note_count']
            logger.info(f"    ✓ Textured MIDI: {stats['note_count']} notes, {stats['duration']:.1f}s")

    # Download MP3 files
    if vocal_chord_mp3:
        mp3_path = song_dir / 'vocal_chord.mp3'
        if download_file(f"{BACKEND_URL}/mp3/{vocal_chord_mp3}", str(mp3_path)):
            logger.info(f"    ✓ Vocal + block MP3 downloaded")

    if vocal_textured_mp3:
        mp3_path = song_dir / 'vocal_textured.mp3'
        if download_file(f"{BACKEND_URL}/mp3/{vocal_textured_mp3}", str(mp3_path)):
            logger.info(f"    ✓ Vocal + textured MP3 downloaded")

    # Estimate chord count from block MIDI (simple heuristic)
    if entry.midi_files.get('chord_gen'):
        try:
            pm = pretty_midi.PrettyMIDI(entry.midi_files['chord_gen'])
            # Count note-on events as a proxy for chord changes
            entry.chord_count = len(pm.instruments[0].notes) if pm.instruments else 0
        except:
            entry.chord_count = 0

    # Save entry as JSON
    entry_json = song_dir / 'analysis.json'
    with open(entry_json, 'w') as f:
        json.dump(asdict(entry), f, indent=2, default=str)

    logger.info(f"  ✓ Results saved to {song_dir}")
    return entry


def download_file(url: str, output_path: str) -> bool:
    """Download file from URL to output_path."""
    try:
        urllib.request.urlretrieve(url, output_path, timeout=30)
        return True
    except Exception as e:
        logger.error(f"  ✗ Download failed: {e}")
        return False


def analyze_midi(path: str) -> dict:
    """Analyze MIDI file and return stats."""
    try:
        pm = pretty_midi.PrettyMIDI(path)
        if not pm.instruments:
            return {'note_count': 0, 'duration': 0.0}

        notes = pm.instruments[0].notes
        duration = max((n.end for n in notes), default=0.0)
        return {'note_count': len(notes), 'duration': duration}
    except Exception as e:
        logger.error(f"  Failed to analyze MIDI: {e}")
        return {'note_count': 0, 'duration': 0.0}


def print_comparison_table(entries: list[UrlAuditEntry]) -> str:
    """Generate Markdown comparison table."""
    lines = [
        "# Baseline Audit Results\n",
        "| Song | Key | Tempo | Chords | Block Notes | Textured Notes | Errors |",
        "|------|-----|-------|--------|-------------|----------------|--------|",
    ]

    for entry in entries:
        key_str = f"{entry.detected_key} {entry.detected_mode}" if entry.detected_key else "?"
        tempo_str = f"{entry.detected_tempo:.0f}" if entry.detected_tempo else "?"
        errors_str = "; ".join(entry.errors[:1]) if entry.errors else "✓"

        line = (
            f"| {entry.song_name} | {key_str} | {tempo_str} | {entry.chord_count} | "
            f"{entry.block_note_count} | {entry.textured_note_count} | {errors_str} |"
        )
        lines.append(line)

    lines.append("\n## Details\n")
    for i, entry in enumerate(entries):
        lines.append(f"### Song {i} ({entry.song_name})\n")
        lines.append(f"- **URL**: {entry.url}")
        lines.append(f"- **Detected Key**: {entry.detected_key} {entry.detected_mode} @ {entry.detected_tempo} BPM")
        lines.append(f"- **Pickup Shift**: {entry.pickup_shift} sixteenths")
        lines.append(f"- **Chord Count**: {entry.chord_count}")
        lines.append(f"- **Block Notes**: {entry.block_note_count} | **Textured Notes**: {entry.textured_note_count}")
        if entry.errors:
            lines.append(f"- **Errors**: {'; '.join(entry.errors)}")
        if entry.midi_files:
            lines.append(f"- **MIDI Files**: {list(entry.midi_files.keys())}")
        if entry.mp3_files:
            vocal_files = [k for k, v in entry.mp3_files.items() if v]
            if vocal_files:
                lines.append(f"- **MP3 Files**: {vocal_files}")
        lines.append("")

    return "\n".join(lines)


def main():
    urls = [
        "https://www.youtube.com/watch?v=falq0Gr3rc0&list=RDfalq0Gr3rc0&start_radio=1",
        "https://www.youtube.com/watch?v=0fcKEN4_QoM&list=RDfalq0Gr3rc0&index=4",
        "https://www.youtube.com/watch?v=b-6uxr6cGQI&list=RDfalq0Gr3rc0&index=22",
        "https://www.youtube.com/watch?v=1hI-7vj2FhE&list=RDfalq0Gr3rc0&index=32",
        "https://www.youtube.com/watch?v=8MG--WuNW1Y&list=RDfalq0Gr3rc0&index=42",
    ]

    logger.info(f"\n{'='*60}")
    logger.info("PHASE 0: BASELINE AUDIT")
    logger.info(f"Backend: {BACKEND_URL}")
    logger.info(f"Engine: rules (ACCOMP_ENGINE=rules)")
    logger.info(f"{'='*60}\n")

    # Clean up old audit results
    if AUDIT_RESULTS_DIR.exists():
        shutil.rmtree(AUDIT_RESULTS_DIR)
    AUDIT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for i, url in enumerate(urls, 1):
        logger.info(f"\nURL {i}/{len(urls)}")
        entry = process_single_url(url)
        results.append(entry)

    # Generate comparison table
    table = print_comparison_table(results)
    table_path = AUDIT_RESULTS_DIR / 'comparison.md'
    with open(table_path, 'w') as f:
        f.write(table)

    logger.info(f"\n{'='*60}")
    logger.info("AUDIT COMPLETE")
    logger.info(f"Results saved to: {AUDIT_RESULTS_DIR}")
    logger.info(f"Comparison table: {table_path}")
    logger.info(f"{'='*60}\n")

    print("\n" + table)


if __name__ == '__main__':
    main()
