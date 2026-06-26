---
title: knee-maze
emoji: 🧭
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: "5.0.0"
app_file: app.py
pinned: false
private: true
---

# knee-maze

Text maze environment + Qwen2.5-1.5B-Instruct agent, running live in a Gradio Space.

**Phase 1:** Baseline loop — generate mazes, run the base model, log trajectories to a private HF Dataset. No training yet.

## Space Secrets

Set these in the Space settings before running:

| Secret | Purpose |
|---|---|
| `HF_TOKEN` | Write access — uploads episode logs to the Dataset repo |
| `DATASET_REPO_ID` | `bolajiev/knee-maze-logs` |

The Qwen base model is downloaded from the public HF Hub automatically (no token needed).

## Local dev

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python test_headless.py   # sanity check, no model download
python app.py             # full Gradio UI
```

`sdk_version` in the frontmatter should match the installed gradio version (`pip show gradio`).
