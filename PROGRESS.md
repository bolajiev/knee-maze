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

**Step 4 — Measure** ✅ Done
| Metric | Base (Phase 1) | Fine-tuned (Phase 2) |
|---|---|---|
| Solve rate | 0% (0/20) | **14.3% (3/21)** |
| Avg steps (wins) | — | 120.7 |
| Avg wall hits/ep | high | **0.0** |
| Timeout rate | 100% | 85.7% |
| Run ID | a3e2b706d5b5 | 711e6f845060 |

**What we learned:**
- Wall hits → 0: model learned to only pick valid moves (SFT worked)
- Still looping: 120 steps to solve vs ~31 BFS-optimal — model wanders valid paths but has no directional strategy
- 14.3% solve rate = wins happen when the model accidentally reaches E before timeout

---

## Phase 3 — DPO ✅ DONE

**Goal:** Use Phase 1 failure logs as preference pairs to make the model avoid its own mistakes.

- **Chosen:** BFS-optimal direction for each maze step
- **Rejected:** valid direction that maximises remaining BFS distance to E (worst informed choice)
- 90k pairs from 3k mazes, trained on 15k, 1 epoch on T4 (~1.3h)
- Starting point: `bolajiev/qwen-maze-sft` (Phase 2 checkpoint)
- Checkpoint: `bolajiev/qwen-maze-dpo`

**DPO training results:**
| Metric | Start | End |
|---|---|---|
| rewards/margins | 0.47 | 2.73 |
| rewards/accuracies | 70.75% | 84% |
| logps/rejected | -9.15 | -33.6 (25× less likely) |

**Step 4 — Measure** ⬜ Not started
- Run 20 episodes with DPO model
- Compare vs 14.3% (SFT) and 0% (base)
- Hypothesis: lower looping = higher solve rate

---

## Key numbers so far
| | |
|---|---|
| Base model solve rate | 0% (20 episodes, 8×8) |
| Phase 2 solve rate | 14.3% (21 episodes, 8×8) |
| SFT training examples used | 20,000 of 309,836 |
| Avg optimal path length | ~31 steps (8×8 maze) |
| Avg steps when fine-tuned wins | 120.7 (4× longer than optimal) |
