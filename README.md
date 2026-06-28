---
title: knee-maze
emoji: 🧭
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: "6.19.0"
app_file: app.py
pinned: false
private: true
hardware: t4-small
---

# knee-maze

RL data pipeline: text maze environment → Qwen2.5-1.5B agent → log trajectories → fine-tune → measure improvement.

**Stack:** Gradio Space (T4 GPU) · HF Dataset for logs · Modal for training · Qwen2.5-1.5B-Instruct

## Phases

| Phase | Goal | Status |
|---|---|---|
| 1 | Baseline loop — run base model, log trajectories | ✅ Done (0% solve rate) |
| 2 | SFT on BFS-optimal paths — beat baseline | ✅ Done (model: bolajiev/qwen-maze-sft) |
| 3 | DPO on failure logs — push higher | ⬜ Planned |

## Space secret required

`HF_TOKEN` — write access to upload episode logs to the Dataset repo.

## Training (Modal)

```bash
pip install modal
modal run train_modal.py
```
