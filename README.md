# Piano accompaniment from YouTube

Pipeline: **YouTube → vocal separation (optional Demucs) → Basic Pitch melody MIDI → tempo / beat hints → LLaMA-MIDI piano accompaniment.**

## Backend

Use a Python 3.10+ env (conda recommended). From the repo root:

```bash
cd back-end
python -m pip install -r requirements.txt
python -m pip install -r requirements-madmom.txt --no-build-isolation
```

Optional madmom improves downbeat alignment; omit that line if you skip it.

Start the API (default `http://127.0.0.1:8765`):

```bash
python app.py
```

First LLaMA-MIDI run downloads the Hugging Face model (~2 GB). Use `python -m pip` if `pip` is not on your PATH.

### Memory (LLaMA-MIDI)

- **Weights:** Llama-3.2-1B is about **2 GB** on disk / in RAM in fp16.
- **During generation:** extra RAM for activations and KV cache grows with **sequence length** (prompt + `LLAMA_MIDI_MAX_NEW_TOKENS`). Plan for roughly **6–10 GB system RAM** for CPU inference at moderate lengths; **NVIDIA GPUs** with **≥6 GB VRAM** are comfortable for this model size.
- **Apple Silicon:** the backend **defaults to CPU** on macOS because **MPS** (Metal) can abort with huge allocations on long generations. Override with `export LLAMA_MIDI_DEVICE=mps` if you want to try the GPU (shorter `LLAMA_MIDI_MAX_NEW_TOKENS` is safer).

## Frontend

```bash
cd front-end
npm install
npm start
```

Production bundle for the Flask `static/` folder:

```bash
./build.sh
```

The HTTP API route is still `/api/chorderator_back_end` for compatibility with the bundled UI.
