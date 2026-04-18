#!/usr/bin/env bash
set -euo pipefail

# Restart helper for this repo version.
# - Runs Flask backend from back-end/app.py
# - Uses .venv Python when available, otherwise falls back to python3/python
# - Optionally rebuilds frontend and copies it to back-end/static via build.sh

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$ROOT_DIR/back-end"
BUILD_SCRIPT="$ROOT_DIR/build.sh"
DEFAULT_PORT="${PORT:-8765}"
DEFAULT_HOST="${HOST:-127.0.0.1}"
APPROVED_SOUNDFONT_DEFAULT="$ROOT_DIR/back-end/assets/soundfonts/MuseScore_General.sf3"
DEFAULT_SOUNDFONT="${APPROVED_SOUNDFONT_PATH:-${SOUNDFONT_PATH:-$APPROVED_SOUNDFONT_DEFAULT}}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-accomontage2}"
REBUILD_UI=1
DAEMON_MODE=0
STOP_ONLY=0
PID_FILE="$BACKEND_DIR/server.pid"
LOG_FILE="$BACKEND_DIR/server.log"

usage() {
  cat <<'EOF'
Usage: ./restart.sh [--rebuild-ui] [--port PORT] [--host HOST] [--daemon] [--stop]

Options:
  --rebuild-ui      Rebuild React frontend and copy to back-end/static (runs ./build.sh)
  --port PORT       Backend port (default: 8765 or $PORT)
  --host HOST       Backend host (default: 127.0.0.1 or $HOST)
  --daemon          Start backend in background and write PID/log files
  --stop            Stop backend listener on selected port and exit
  -h, --help        Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rebuild-ui)
      REBUILD_UI=1
      shift
      ;;
    --port)
      if [[ $# -lt 2 ]]; then
        echo "Error: --port requires a value" >&2
        exit 1
      fi
      DEFAULT_PORT="$2"
      shift 2
      ;;
    --host)
      if [[ $# -lt 2 ]]; then
        echo "Error: --host requires a value" >&2
        exit 1
      fi
      DEFAULT_HOST="$2"
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

if [[ ! -d "$BACKEND_DIR" ]]; then
  echo "Error: back-end directory not found at $BACKEND_DIR" >&2
  exit 1
fi

if [[ "$REBUILD_UI" -eq 1 ]]; then
  if [[ ! -x "$BUILD_SCRIPT" ]]; then
    chmod +x "$BUILD_SCRIPT"
  fi
  echo "Rebuilding frontend and syncing static assets..."
  "$BUILD_SCRIPT"
fi

# Pick Python runtime; prefer requested conda env when available.
PYTHON_CMD=""
PYTHON_ARGS=()
CONDA_BIN=""
if command -v conda >/dev/null 2>&1; then
  CONDA_BIN="$(command -v conda)"
elif [[ -x "/opt/anaconda3/bin/conda" ]]; then
  CONDA_BIN="/opt/anaconda3/bin/conda"
fi

if [[ -n "$CONDA_BIN" ]]; then
  # Prefer the env's python executable directly to avoid extra conda-run wrapper noise.
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

# Stop any running Python listener on the same port first.
if command -v lsof >/dev/null 2>&1; then
  PIDS="$(lsof -tiTCP:"$DEFAULT_PORT" -sTCP:LISTEN 2>/dev/null || true)"
  if [[ -n "$PIDS" ]]; then
    echo "Stopping existing process(es) on port $DEFAULT_PORT: $PIDS"
    while read -r PID; do
      [[ -z "$PID" ]] && continue
      COMM="$(ps -p "$PID" -o comm= 2>/dev/null | tr -d '[:space:]' || true)"
      if [[ "$COMM" == *python* ]]; then
        kill "$PID" 2>/dev/null || true
      fi
    done <<< "$PIDS"

    # Brief pause so the socket is released.
    sleep 1
  fi
fi

if [[ "$STOP_ONLY" -eq 1 ]]; then
  if [[ -f "$PID_FILE" ]]; then
    rm -f "$PID_FILE"
  fi
  echo "Stopped listeners on port $DEFAULT_PORT"
  exit 0
fi

cd "$BACKEND_DIR"

if [[ ${#PYTHON_ARGS[@]} -gt 0 ]]; then
  echo "Starting backend with: $PYTHON_CMD ${PYTHON_ARGS[*]} app.py"
else
  echo "Starting backend with: $PYTHON_CMD app.py"
fi
echo "Host: $DEFAULT_HOST"
echo "Port: $DEFAULT_PORT"
echo "URL : http://$DEFAULT_HOST:$DEFAULT_PORT"

export HOST="$DEFAULT_HOST"
export PORT="$DEFAULT_PORT"

if [[ -n "$DEFAULT_SOUNDFONT" ]]; then
  export SOUNDFONT_PATH="$DEFAULT_SOUNDFONT"
  echo "Soundfont: $SOUNDFONT_PATH"
else
  echo "Soundfont: none selected (set APPROVED_SOUNDFONT_PATH after you verify a preview)"
fi

if [[ "$DAEMON_MODE" -eq 1 ]]; then
  echo "Daemon log: $LOG_FILE"
  echo "Daemon pid: $PID_FILE"
  if [[ ${#PYTHON_ARGS[@]} -gt 0 ]]; then
    nohup "$PYTHON_CMD" "${PYTHON_ARGS[@]}" app.py >"$LOG_FILE" 2>&1 &
  else
    nohup "$PYTHON_CMD" app.py >"$LOG_FILE" 2>&1 &
  fi
  echo $! > "$PID_FILE"
  echo "Started daemon PID $(cat "$PID_FILE")"
  exit 0
fi

if [[ ${#PYTHON_ARGS[@]} -gt 0 ]]; then
  exec "$PYTHON_CMD" "${PYTHON_ARGS[@]}" app.py
else
  exec "$PYTHON_CMD" app.py
fi
