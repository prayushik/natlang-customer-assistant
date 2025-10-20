# Natlang – Assignment 6 Code Bundle

## Quick Start (Windows PowerShell or VS Code Terminal)
```powershell
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:OPENAI_API_KEY="sk-...yourkey..."   # optional; falls back to local composer if not set
python natlang.py
```
Screenshots and JSON will be saved to `./natlang_artifacts/`.

## Quick Start (macOS/Linux)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="sk-...yourkey..."   # optional; falls back if not set
python natlang.py
```

## Files
- `natlang.py` – full pipeline (OpenAI call with fallback) + demo for 5 examples
- `requirements.txt` – dependencies
- `run_demo.bat` / `run_demo.sh` – one-step helpers
- Output folder: `natlang_artifacts/` (created on first run)

## Notes
- If `OPENAI_API_KEY` is not set, the app uses a deterministic fallback composer, still saving PNGs and JSON.
- Replace the five example strings in `demo()` if you want to test custom scenarios.
