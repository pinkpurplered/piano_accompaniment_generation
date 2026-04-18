import io
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import logging
import time

import pretty_midi
from flask import Flask, request, send_from_directory, send_file, make_response, jsonify

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import chorderator as cdt
from Sessions import Sessions
import melody_analyze
import youtube_melody
import audio_mixer

app = Flask(__name__, static_url_path='')
app.secret_key = 'AccoMontage2-GUI'
saved_data = cdt.load_data()
APP_ROUTE = '/api/chorderator_back_end'
sessions = Sessions()
logging.basicConfig(level=logging.DEBUG)
MIDI_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'midi')
MP3_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'mp3')
os.makedirs(MIDI_FOLDER, exist_ok=True)
os.makedirs(MP3_FOLDER, exist_ok=True)

# File cleanup configuration
CLEANUP_MAX_AGE_SECONDS = 3600  # 1 hour - files older than this will be deleted


def cleanup_old_files():
    """Remove old generated files and session directories to save disk space."""
    try:
        current_time = time.time()
        base_dir = os.path.dirname(__file__)
        
        # Clean up old MIDI files in static/midi
        for filename in os.listdir(MIDI_FOLDER):
            if filename.endswith('.mid'):
                filepath = os.path.join(MIDI_FOLDER, filename)
                try:
                    if os.path.isfile(filepath):
                        file_age = current_time - os.path.getmtime(filepath)
                        if file_age > CLEANUP_MAX_AGE_SECONDS:
                            os.remove(filepath)
                            logging.info(f'Cleaned up old MIDI file: {filename}')
                except Exception as e:
                    logging.warning(f'Failed to clean up MIDI file {filename}: {e}')
        
        # Clean up old MP3 files in static/mp3
        for filename in os.listdir(MP3_FOLDER):
            if filename.endswith('.mp3'):
                filepath = os.path.join(MP3_FOLDER, filename)
                try:
                    if os.path.isfile(filepath):
                        file_age = current_time - os.path.getmtime(filepath)
                        if file_age > CLEANUP_MAX_AGE_SECONDS:
                            os.remove(filepath)
                            logging.info(f'Cleaned up old MP3 file: {filename}')
                except Exception as e:
                    logging.warning(f'Failed to clean up MP3 file {filename}: {e}')
        
        # Clean up old session directories (UUID-format directories older than cleanup threshold)
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            if os.path.isdir(item_path) and '-' in item and len(item) == 36:
                try:
                    # Check if it's a UUID-format directory
                    dir_age = current_time - os.path.getmtime(item_path)
                    if dir_age > CLEANUP_MAX_AGE_SECONDS:
                        shutil.rmtree(item_path, ignore_errors=True)
                        logging.info(f'Cleaned up old session directory: {item}')
                except Exception as e:
                    logging.warning(f'Failed to clean up session directory {item}: {e}')
                    
    except Exception as e:
        logging.error(f'Cleanup failed: {e}')


def session_from_request(req):
    """(session, session_id) or (None, None) if cookie missing or unknown (never unpack bare get_session)."""
    p = sessions.get_session(req)
    return (None, None) if p is None else p


def resp(msg=None, session_id=None, more=()):
    body = {'status': 'ok' if not msg else msg}
    for item in more:
        body[item[0]] = item[1]
    r = make_response(jsonify(body))
    if session_id:
        r.set_cookie('session', session_id, max_age=3600)
    return r


def send_file_from_session(file, name=None):
    return send_file(
        io.BytesIO(file),
        as_attachment=True,
        download_name=name,
    )


def extract_accompaniment_only(midi_obj):
    """
    Extract only accompaniment tracks from MIDI, removing the melody.
    Assumes the melody is the first instrument (track 0) or the monophonic lead.
    """
    new_midi = pretty_midi.PrettyMIDI()
    new_midi.time_signature_changes = midi_obj.time_signature_changes
    new_midi.key_signature_changes = midi_obj.key_signature_changes
    
    # Copy only non-melody instruments (skip first track which is typically the melody)
    if len(midi_obj.instruments) > 1:
        for instrument in midi_obj.instruments[1:]:
            new_midi.instruments.append(instrument)
    else:
        # If there's only one instrument, return empty MIDI with proper structure
        logging.warning("MIDI has only one instrument - likely melody only")
        
    return new_midi


def begin_generate_thread(core, session_id):
    try:
        core.generate_save(session_id, log=True)
    except Exception as e:
        logging.exception("generate_save failed session=%s", session_id)
        if session_id in sessions.sessions:
            sessions.sessions[session_id].generate_error = (
                f"{type(e).__name__}: {e}. "
                "Often caused by a phrase slice length the chord library does not cover "
                "(e.g. 12 bars in the last segment). Use even 8-bar phrases such as A8B8A8B8, "
                "or trim the MIDI grid."
            )[:900]
        try:
            core.state = 7
        except Exception:
            pass


@app.route(APP_ROUTE + '/upload_youtube', methods=['POST'])
def upload_youtube():
    # Clean up old files periodically
    cleanup_old_files()
    
    data = request.get_json(silent=True) or {}
    url = (data.get('url') or '').strip()
    use_vocal_only = bool(data.get('use_vocal_only', True))
    if not url:
        return resp(msg='missing url')
    if sessions.get_session(request) is None:
        session, session_id = sessions.create_session()
        logging.debug('create new session {}'.format(session_id))
    else:
        session, session_id = sessions.get_session(request)
        logging.debug('request is in session {}'.format(session_id))
    work_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), session_id)
    try:
        midi_bytes, hints = youtube_melody.youtube_url_to_midi_bytes(url, work_dir, use_vocal_only=use_vocal_only)
        session.melody = midi_bytes
        
        # Log whether vocals were created for future MP3 generation
        vocals_path = os.path.join(work_dir, "demucs_stems")
        if os.path.isdir(vocals_path):
            # Check if vocals.wav exists
            vocals_found = False
            for root, dirs, files in os.walk(vocals_path):
                if "vocals.wav" in files:
                    vocals_found = True
                    logging.info("✓ Demucs vocals created and available for MP3 generation: %s", os.path.join(root, "vocals.wav"))
                    break
            if not vocals_found:
                logging.warning("⚠️ demucs_stems directory exists but vocals.wav not found - MP3 generation will fail!")
        else:
            logging.warning("⚠️ No demucs_stems directory - vocals were not separated. MP3 generation will attempt separation later.")
            
    except Exception as e:
        logging.exception('upload_youtube failed')
        return resp(msg=str(e)[:900])
    analysis = melody_analyze.analyze_melody_bytes(session.melody, key_hints=hints)
    if analysis.get("detected_tempo") is not None:
        session.tempo = analysis.get("detected_tempo")
    return resp(
        session_id=session_id,
        more=melody_analyze.build_response_more(hints, analysis),
    )


@app.route(APP_ROUTE + '/generate', methods=['POST'])
def generate():
    session, session_id = session_from_request(request)
    if session is None:
        return resp(msg='session expired')
    if not session.melody:
        return resp(msg='load melody first (use YouTube Load melody on the previous step)')
    logging.debug('request is in session {}'.format(session_id))
    params = json.loads(request.data)
    session.load_params(params)
    if not session.tempo:
        session.tempo = 120
    session.generate_error = None

    logging.info(
        'generate meta tonic=%s mode=%s meter=%s tempo=%s segmentation=%s',
        session.tonic,
        session.mode,
        session.meter,
        session.tempo,
        session.segmentation,
    )

    session.core = cdt.get_chorderator()
    session.core.set_cache(**saved_data)
    
    # Use absolute path for session directory
    session_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), session_id)
    os.makedirs(session_dir, exist_ok=True)
    
    full_song_path = os.path.join(session_dir, 'full_song.mid')
    with open(full_song_path, 'wb') as f:
        f.write(session.melody)
    session.core.set_melody(full_song_path)
    session.core.set_output_style(session.chord_style)
    session.core.set_texture_prefilter(session.texture_style)
    session.core.set_meta(tonic=session.tonic, meter=session.meter, mode=session.mode, tempo=session.tempo)
    session.core.set_segmentation(session.segmentation)
    threading.Thread(target=begin_generate_thread, args=(session.core, session_id)).start()
    return resp(session_id=session_id)


@app.route(APP_ROUTE + '/stage_query', methods=['GET'])
def answer_stage():
    session, session_id = session_from_request(request)
    if session is None:
        return resp(msg='session expired')
    logging.debug('request is in session {}'.format(session_id))
    if getattr(session, "generate_error", None):
        return resp(
            session_id=session_id,
            more=[
                ["stage", "0"],
                ["generate_error", session.generate_error],
            ],
        )
    return resp(session_id=session_id, more=[['stage', str(session.core.get_state())]])


@app.route(APP_ROUTE + '/generated_query', methods=['GET'])
def answer_gen():
    session, session_id = session_from_request(request)
    if session is None:
        return resp(msg='session expired')
    logging.debug('request is in session {}'.format(session_id))
    
    chord_midi_name = session_id + '_' + str(time.time()) + '_chord_gen.mid'
    acc_midi_name = session_id + '_' + str(time.time()) + '_textured_chord_gen.mid'
    
    # Construct full path to session directory
    session_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), session_id)
    
    # Copy MIDI files to static folder, extracting only accompaniment tracks
    chord_gen_path = os.path.join(session_dir, 'chord_gen.mid')
    if os.path.exists(chord_gen_path):
        midi_obj = pretty_midi.PrettyMIDI(chord_gen_path)
        # Remove melody track (typically the first instrument or highest-pitched monophonic track)
        accompaniment_only = extract_accompaniment_only(midi_obj)
        accompaniment_only.write(os.path.join(MIDI_FOLDER, chord_midi_name))
    
    textured_gen_path = os.path.join(session_dir, 'textured_chord_gen.mid')
    if os.path.exists(textured_gen_path):
        midi_obj = pretty_midi.PrettyMIDI(textured_gen_path)
        accompaniment_only = extract_accompaniment_only(midi_obj)
        accompaniment_only.write(os.path.join(MIDI_FOLDER, acc_midi_name))
    
    # Generate MP3 mixes
    mp3_names = {}
    try:
        logging.info('MP3 generation: work_dir=%s', session_dir)
        
        mp3_names = audio_mixer.create_mixed_outputs(
            session_id=session_id,
            work_dir=session_dir,
            output_mp3_dir=MP3_FOLDER,
        )
        logging.info('MP3 files generated: %s', mp3_names)
    except Exception as e:
        logging.error('MP3 generation failed: %s', e)
        # Continue even if MP3 generation fails - at least return MIDI files
        mp3_names = {"vocal_chord_mp3": None, "vocal_textured_mp3": None}
    
    # Clean up session directory
    shutil.rmtree(session_dir, ignore_errors=True)
    
    # Clean up old static files and session directories
    cleanup_old_files()
    
    return resp(session_id=session_id,
                more=[
                    ['chord_midi_name', chord_midi_name],
                    ['acc_midi_name', acc_midi_name],
                    ['vocal_chord_mp3', mp3_names.get('vocal_chord_mp3')],
                    ['vocal_textured_mp3', mp3_names.get('vocal_textured_mp3')],
                ])


@app.route(APP_ROUTE + '/midi/<ran>', methods=['GET'])
def midi(ran):
    safe_name = os.path.basename(ran)
    if not safe_name.endswith('.mid'):
        return resp(msg='invalid midi name')
    midi_path = os.path.join(MIDI_FOLDER, safe_name)
    if not os.path.isfile(midi_path):
        return resp(msg='midi not found')
    return send_from_directory(MIDI_FOLDER, safe_name, as_attachment=True, download_name=safe_name)


@app.route(APP_ROUTE + '/mp3/<ran>', methods=['GET'])
def mp3(ran):
    safe_name = os.path.basename(ran)
    if not safe_name.endswith('.mp3'):
        return resp(msg='invalid mp3 name')
    mp3_path = os.path.join(MP3_FOLDER, safe_name)
    if not os.path.isfile(mp3_path):
        return resp(msg='mp3 not found')
    return send_from_directory(MP3_FOLDER, safe_name, mimetype='audio/mpeg')


@app.errorhandler(404)
def index(error):
    return make_response(send_from_directory('static', 'index.html'))


def _kill_python_listeners_on_port(port):
    """SIGTERM stale Flask/Python listeners so restart works. Skips non-Python (e.g. AirPlay)."""
    try:
        r = subprocess.run(
            ['lsof', '-i', f':{port}', '-sTCP:LISTEN', '-n', '-P', '-t'],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return
    for pid_s in r.stdout.strip().split():
        try:
            pid = int(pid_s)
        except ValueError:
            continue
        try:
            comm = subprocess.check_output(
                ['ps', '-p', str(pid), '-o', 'comm='],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except subprocess.CalledProcessError:
            continue
        base = os.path.basename((comm or '').split()[0]).lower()
        if 'python' not in base:
            continue
        try:
            os.kill(pid, signal.SIGTERM)
            logging.info('stopped prior Python listener pid=%s on port %s', pid, port)
            time.sleep(0.25)
        except ProcessLookupError:
            pass
        except PermissionError:
            logging.warning('could not stop pid=%s on port %s (permission)', pid, port)


if __name__ == '__main__':
    # Default avoids macOS AirPlay Receiver (listens on :5000; returns HTTP 403 as AirTunes).
    port = int(os.environ.get('PORT', '8765'))
    host = os.environ.get('HOST', '127.0.0.1')
    if sys.platform == 'darwin' and port == 5000:
        logging.warning(
            'macOS: port 5000 is often taken by AirPlay (AirTunes) and returns HTTP 403 in the browser. '
            'Prefer the default port 8765, or use http://127.0.0.1:5000/ only if AirPlay is off, '
            'or set PORT to another free port.',
        )
    if os.environ.get('SKIP_FREE_PORT', '').lower() not in ('1', 'true', 'yes'):
        _kill_python_listeners_on_port(port)
    
    # Clean up old files on startup
    logging.info('Cleaning up old generated files on startup...')
    cleanup_old_files()
    
    app.run(host=host, port=port)
