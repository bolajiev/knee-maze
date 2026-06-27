# knee-maze — Project Progress

## What this project is
Fine-tuning Qwen2.5-1.5B to navigate text mazes using RL-style data collection.
The loop: baseline → collect data → fine-tune → measure improvement → repeat.

---

## Phase 1 — Baseline ✅ DONE

**Goal:** Prove the pipeline works. Get a baseline solve rate for the base model.

**Built:**
- Procedural maze generator (randomized recursive backtracker, deterministic per seed)
- Text renderer (`#` walls, `.` paths, `S` start, `E` end, `@` agent)
- `model_agent` — Qwen2.5-1.5B-Instruct loaded in-process via `transformers`
- Live Gradio UI — two panels (Base / Fine-tuned), step-by-step colored maze rendering
- Per-step + per-episode JSONL logging → pushed to private HF Dataset repo
- BFS solver for maze (used in Phase 2 dataset generation)

**Infrastructure:**
- HF Space: `bolajiev/knee-maze` (private, Gradio SDK, T4 Small GPU)
- HF Dataset: `bolajiev/knee-maze-logs` (logs + SFT training data)
- GitHub: `github.com/bolajiev/knee-maze`

**Baseline result:**
| Metric | Value |
|---|---|
| Model | Qwen2.5-1.5B-Instruct (base, no fine-tuning) |
| Maze size | 8×8 |
| Episodes | 20 |
| Solve rate | **0%** |
| Timeout rate | 100% |
| Run ID | a3e2b706d5b5 |

Base model cannot navigate an 8×8 maze at all. Picks valid moves but loops — no strategy.

---

## Phase 2 — SFT Fine-tuning ✅ DONE

**Goal:** Fine-tune on BFS-optimal trajectories. Beat 0% baseline.

**Step 1 — Generate dataset** ✅ Done
- Script: `generate_dataset.py`
- 10,000 mazes × 31 steps avg = **309,836 training examples** (177MB)
- Format: chat SFT — user gets maze grid + valid moves, assistant outputs optimal direction
- Uploaded: `bolajiev/knee-maze-logs/sft/train.jsonl`

**Step 2 — Fine-tune on Modal** ✅ Done
- Script: `train_modal.py`
- LoRA SFT on T4 GPU — ran ~1.94 hours
- fp16, device_map={"": 0}, 20k examples, 3 epochs
- Final loss: 0.226 | Token accuracy: 90.5%
- Checkpoint: `bolajiev/qwen-maze-sft` (private HF model repo)

**Step 3 — Activate fine-tuned panel** ✅ Done
- `config.py`: `FINE_TUNED_MODEL_PATH = "bolajiev/qwen-maze-sft"`
- Pushed to HF Space → right panel now live

**Step 4 — Measure** ⬜ Not started
- Run 20 episodes with fine-tuned model
- Compare solve rate vs 0% baseline
- Target: >50% (if not, investigate before Phase 3)

---

## Phase 3 — DPO ⬜ NOT STARTED

**Goal:** Use Phase 1 failure logs as preference pairs to make the model avoid its own mistakes.

- **Chosen:** BFS-optimal action for each maze state
- **Rejected:** `intended_action` where `wall_hit: true` from Phase 1 logs
- Trains on top of Phase 2 SFT checkpoint, not base model
- Expected to push solve rate higher than Phase 2 alone

---

## Key numbers so far
| | |
|---|---|
| Base model solve rate | 0% (20 episodes, 8×8) |
| Phase 2 target | >50% |
| SFT training examples | ~310,000 |
| Avg optimal path length | ~31 steps (8×8 maze) |
