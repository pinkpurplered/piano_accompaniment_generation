#!/usr/bin/env bash
set -euo pipefail

# Restart backend (Flask) and frontend (CRA dev server).
# - Backend: back-end/app.py (default http://127.0.0.1:8765)
# - Frontend: front-end/npm start (default http://localhost:3000, proxies /api)
#
# Environment: CONDA_ENV_NAME (default midi), PORT, HOST, FRONTEND_PORT

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$ROOT_DIR/back-end"
FRONTEND_DIR="$ROOT_DIR/front-end"
BUILD_SCRIPT="$ROOT_DIR/build.sh"
DEFAULT_PORT="${PORT:-8765}"
DEFAULT_HOST="${HOST:-127.0.0.1}"
DEFAULT_FRONTEND_PORT="${FRONTEND_PORT:-3000}"
APPROVED_SOUNDFONT_DEFAULT="$ROOT_DIR/back-end/assets/soundfonts/MuseScore_General.sf3"
DEFAULT_SOUNDFONT="${APPROVED_SOUNDFONT_PATH:-${SOUNDFONT_PATH:-$APPROVED_SOUNDFONT_DEFAULT}}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-midi}"

REBUILD_UI=0
DAEMON_MODE=0
STOP_ONLY=0
BACKEND_ONLY=0
FRONTEND_ONLY=0

BACKEND_PID_FILE="$BACKEND_DIR/server.pid"
BACKEND_LOG_FILE="$BACKEND_DIR/server.log"
FRONTEND_PID_FILE="$FRONTEND_DIR/dev-server.pid"
FRONTEND_LOG_FILE="$FRONTEND_DIR/dev-server.log"

usage() {
  cat <<'EOF'
Usage: ./restart.sh [options]

Starts both the Flask API and the React dev server unless you pass --backend-only
or --frontend-only.

Options:
  --rebuild-ui       Run ./build.sh (production bundle → back-end/static)
  --port PORT        Backend port (default: 8765 or $PORT)
  --host HOST        Backend bind host (default: 127.0.0.1 or $HOST)
  --frontend-port P  Dev server port (default: 3000 or $FRONTEND_PORT)
  --daemon           Run backend and frontend in background; write pid/log files
  --stop             Stop listeners on backend + frontend ports and exit
  --backend-only     Only restart Flask (no npm)
  --frontend-only    Only restart CRA (assumes backend already running)
  -h, --help         Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rebuild-ui)
      REBUILD_UI=1
      shift
      ;;
    --port)
      [[ $# -lt 2 ]] && { echo "Error: --port requires a value" >&2; exit 1; }
      DEFAULT_PORT="$2"
      shift 2
      ;;
    --host)
      [[ $# -lt 2 ]] && { echo "Error: --host requires a value" >&2; exit 1; }
      DEFAULT_HOST="$2"
      shift 2
      ;;
    --frontend-port)
      [[ $# -lt 2 ]] && { echo "Error: --frontend-port requires a value" >&2; exit 1; }
      DEFAULT_FRONTEND_PORT="$2"
      shift 2
      ;;
    --daemon)
      DAEMON_MODE=1
      shift
      ;;
    --stop)
      STOP_ONLY=1
      shift
      ;;
    --backend-only)
      BACKEND_ONLY=1
      shift
      ;;
    --frontend-only)
      FRONTEND_ONLY=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ "$BACKEND_ONLY" -eq 1 && "$FRONTEND_ONLY" -eq 1 ]]; then
  echo "Error: use only one of --backend-only or --frontend-only" >&2
  exit 1
fi

if [[ ! -d "$BACKEND_DIR" ]]; then
  echo "Error: back-end directory not found at $BACKEND_DIR" >&2
  exit 1
fi

if [[ "$FRONTEND_ONLY" -eq 0 && ! -d "$FRONTEND_DIR" ]]; then
  echo "Error: front-end directory not found at $FRONTEND_DIR" >&2
  exit 1
fi

if [[ "$REBUILD_UI" -eq 1 ]]; then
  if [[ ! -x "$BUILD_SCRIPT" ]]; then
    chmod +x "$BUILD_SCRIPT"
  fi
  echo "Rebuilding frontend and syncing static assets..."
  "$BUILD_SCRIPT"
fi

# --- Python for backend ---
PYTHON_CMD=""
PYTHON_ARGS=()
CONDA_BIN=""
if command -v conda >/dev/null 2>&1; then
  CONDA_BIN="$(command -v conda)"
elif [[ -x "/opt/anaconda3/bin/conda" ]]; then
  CONDA_BIN="/opt/anaconda3/bin/conda"
fi

if [[ -n "$CONDA_BIN" ]]; then
  CONDA_ENV_PY=""
  if [[ -x "/opt/anaconda3/envs/$CONDA_ENV_NAME/bin/python" ]]; then
    CONDA_ENV_PY="/opt/anaconda3/envs/$CONDA_ENV_NAME/bin/python"
  else
    CONDA_PREFIX_LINE="$($CONDA_BIN info --envs | awk -v env="$CONDA_ENV_NAME" '$1==env {print $NF}' | head -1 || true)"
    if [[ -n "$CONDA_PREFIX_LINE" && -x "$CONDA_PREFIX_LINE/bin/python" ]]; then
      CONDA_ENV_PY="$CONDA_PREFIX_LINE/bin/python"
    fi
  fi
  if [[ -n "$CONDA_ENV_PY" ]]; then
    PYTHON_CMD="$CONDA_ENV_PY"
  elif "$CONDA_BIN" run -n "$CONDA_ENV_NAME" python -c "import sys" >/dev/null 2>&1; then
    PYTHON_CMD="$CONDA_BIN"
    PYTHON_ARGS=(run --no-capture-output -n "$CONDA_ENV_NAME" python)
  fi
fi

if [[ -z "$PYTHON_CMD" ]]; then
  if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
    PYTHON_CMD="$ROOT_DIR/.venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_CMD="$(command -v python)"
  else
    echo "Error: no Python executable found (conda env, .venv/bin/python, python3, or python)." >&2
    exit 1
  fi
fi

stop_port_listeners() {
  local port="$1"
  local match_substr="$2" # e.g. python or node — only kill if comm matches
  if ! command -v lsof >/dev/null 2>&1; then
    echo "Warning: lsof not found; cannot free port $port" >&2
    return 0
  fi
  local PIDS
  PIDS="$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
  if [[ -z "$PIDS" ]]; then
    return 0
  fi
  echo "Stopping process(es) on port $port: $PIDS"
  while read -r PID; do
    [[ -z "$PID" ]] && continue
    local COMM
    COMM="$(ps -p "$PID" -o comm= 2>/dev/null | tr -d '[:space:]' || true)"
    if [[ "$COMM" == *"$match_substr"* ]]; then
      kill "$PID" 2>/dev/null || true
    else
      echo "  (skip pid $PID comm=$COMM — not *$match_substr*)" >&2
    fi
  done <<< "$PIDS"
  sleep 1

  # If anything still listens on this port after SIGTERM, force stop it.
  local REMAINING
  REMAINING="$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
  if [[ -n "$REMAINING" ]]; then
    echo "Force-stopping remaining listener(s) on port $port: $REMAINING"
    while read -r PID; do
      [[ -z "$PID" ]] && continue
      local COMM
      COMM="$(ps -p "$PID" -o comm= 2>/dev/null | tr -d '[:space:]' || true)"
      if [[ "$COMM" == *"$match_substr"* ]]; then
        kill -9 "$PID" 2>/dev/null || true
      fi
    done <<< "$REMAINING"
  fi
}

if [[ "$FRONTEND_ONLY" -eq 0 ]]; then
  stop_port_listeners "$DEFAULT_PORT" "python"
fi
if [[ "$BACKEND_ONLY" -eq 0 ]]; then
  stop_port_listeners "$DEFAULT_FRONTEND_PORT" "node"
fi

rm -f "$BACKEND_PID_FILE" "$FRONTEND_PID_FILE"

if [[ "$STOP_ONLY" -eq 1 ]]; then
  echo "Stopped listeners on backend port $DEFAULT_PORT and frontend port $DEFAULT_FRONTEND_PORT"
  exit 0
fi

if [[ "$FRONTEND_ONLY" -eq 0 ]]; then
  if [[ ${#PYTHON_ARGS[@]} -gt 0 ]]; then
    echo "Backend Python: $PYTHON_CMD ${PYTHON_ARGS[*]}"
  else
    echo "Backend Python: $PYTHON_CMD"
  fi
  echo "Backend URL: http://$DEFAULT_HOST:$DEFAULT_PORT"
fi
if [[ "$BACKEND_ONLY" -eq 0 ]]; then
  if ! command -v npm >/dev/null 2>&1; then
    echo "Error: npm not found; install Node.js or use --backend-only" >&2
    exit 1
  fi
  echo "Frontend dev: http://127.0.0.1:$DEFAULT_FRONTEND_PORT (PORT=$DEFAULT_FRONTEND_PORT npm start)"
fi

export HOST="$DEFAULT_HOST"
export PORT="$DEFAULT_PORT"
if [[ -n "$DEFAULT_SOUNDFONT" ]]; then
  export SOUNDFONT_PATH="$DEFAULT_SOUNDFONT"
  echo "Soundfont: $SOUNDFONT_PATH"
fi

# LLaMA-MIDI generation quality settings (optimized for emotional ballad piano accompaniment)
# Higher temperature (1.0) = more creative and expressive, follows melody better
# Max tokens (16000) = generates FULL-LENGTH accompaniment
# CRITICAL: Must be high enough to prevent looping the same bar repeatedly!
export LLAMA_MIDI_MAX_NEW_TOKENS="${LLAMA_MIDI_MAX_NEW_TOKENS:-16000}"
export LLAMA_MIDI_TEMPERATURE="${LLAMA_MIDI_TEMPERATURE:-1.0}"
export LLAMA_MIDI_TOP_P="${LLAMA_MIDI_TOP_P:-0.92}"

# Device selection (CPU is recommended for MacBook Air and stable long generations)
# cpu = Slower (~5-10 min/song) but reliable, no memory issues
# mps = Faster (~1-2 min/song) but may crash with "out of memory" on long songs
export LLAMA_MIDI_DEVICE="${LLAMA_MIDI_DEVICE:-cpu}"

echo "LLaMA-MIDI: max_tokens=$LLAMA_MIDI_MAX_NEW_TOKENS temp=$LLAMA_MIDI_TEMPERATURE top_p=$LLAMA_MIDI_TOP_P device=$LLAMA_MIDI_DEVICE"
if [[ "$LLAMA_MIDI_DEVICE" == "cpu" ]]; then
  echo "  Using CPU mode (recommended for MacBook Air) - generation takes ~5-10 minutes per song"
  echo "  This is slower but stable with no memory issues"
elif [[ "$LLAMA_MIDI_DEVICE" == "mps" ]]; then
  echo "  ⚡ Using Apple Metal GPU - faster but may crash on long songs"
  echo "  If you get 'out of memory' errors, the server will switch to CPU automatically"
fi

if [[ "$DAEMON_MODE" -eq 1 ]]; then
  if [[ "$FRONTEND_ONLY" -eq 0 ]]; then
    echo "Backend log: $BACKEND_LOG_FILE"
    pushd "$BACKEND_DIR" >/dev/null
    if [[ ${#PYTHON_ARGS[@]} -gt 0 ]]; then
      nohup "$PYTHON_CMD" "${PYTHON_ARGS[@]}" app.py >"$BACKEND_LOG_FILE" 2>&1 &
    else
      nohup "$PYTHON_CMD" app.py >"$BACKEND_LOG_FILE" 2>&1 &
    fi
    echo $! >"$BACKEND_PID_FILE"
    popd >/dev/null
    echo "Backend daemon PID $(cat "$BACKEND_PID_FILE")"
  fi
  if [[ "$BACKEND_ONLY" -eq 0 ]]; then
    echo "Frontend log: $FRONTEND_LOG_FILE"
    pushd "$FRONTEND_DIR" >/dev/null
    PORT="$DEFAULT_FRONTEND_PORT" BROWSER="${BROWSER:-none}" \
      nohup npm start >"$FRONTEND_LOG_FILE" 2>&1 &
    echo $! >"$FRONTEND_PID_FILE"
    popd >/dev/null
    echo "Frontend daemon PID $(cat "$FRONTEND_PID_FILE")"
  fi
  exit 0
fi

# Foreground: one process must own the TTY; run frontend in foreground and backend in background.
if [[ "$FRONTEND_ONLY" -eq 1 ]]; then
  exec bash -c "cd \"$FRONTEND_DIR\" && export PORT=\"$DEFAULT_FRONTEND_PORT\" BROWSER=\"${BROWSER:-none}\" && exec npm start"
fi

if [[ "$BACKEND_ONLY" -eq 1 ]]; then
  cd "$BACKEND_DIR"
  if [[ ${#PYTHON_ARGS[@]} -gt 0 ]]; then
    exec "$PYTHON_CMD" "${PYTHON_ARGS[@]}" app.py
  else
    exec "$PYTHON_CMD" app.py
  fi
fi

BACKEND_BG_PID=""
(
  cd "$BACKEND_DIR"
  if [[ ${#PYTHON_ARGS[@]} -gt 0 ]]; then
    exec "$PYTHON_CMD" "${PYTHON_ARGS[@]}" app.py
  else
    exec "$PYTHON_CMD" app.py
  fi
) &
BACKEND_BG_PID=$!

cleanup_backend() {
  if [[ -n "$BACKEND_BG_PID" ]] && kill -0 "$BACKEND_BG_PID" 2>/dev/null; then
    kill "$BACKEND_BG_PID" 2>/dev/null || true
    wait "$BACKEND_BG_PID" 2>/dev/null || true
  fi
}
trap cleanup_backend EXIT INT TERM

cd "$FRONTEND_DIR"
export PORT="$DEFAULT_FRONTEND_PORT"
export BROWSER="${BROWSER:-none}"
npm start
trap - EXIT INT TERM
