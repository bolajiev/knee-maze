# knee-maze — Phase 2 Spec: SFT Fine-tuning

**Project:** `knee-maze`
**Phase:** 2 — Supervised fine-tuning on BFS-optimal trajectories
**Input:** Baseline solve-rate from Phase 1 (Qwen2.5-1.5B on 8x8 mazes, no training)
**Goal:** Fine-tune Qwen2.5-1.5B on programmatically-generated optimal maze solutions, then compare solve rate against the Phase 1 baseline using the existing Space UI.

---

## 1. The Core Insight

Phase 1 logs alone are not enough. With a ~5-10% baseline solve rate, 50 episodes yields maybe 3-5 winning trajectories — nowhere near enough signal for SFT.

The maze has a computable optimal solution via BFS. We don't need to wait for the model to accidentally solve anything. We generate a solver, run it on thousands of random mazes, and produce unlimited clean `(maze_state → correct_direction)` training pairs. The Phase 1 logs become Phase 3 material (DPO preference pairs). Phase 2 is purely SFT on BFS gold data.

---

## 2. Scope

**In scope:**
- BFS solver for the existing `Maze` object
- Dataset generator: runs solver on N mazes, formats as SFT chat examples, pushes to HF Dataset
- Fine-tuning script on Modal (GPU): LoRA SFT via `trl.SFTTrainer`
- Upload fine-tuned checkpoint to HF Hub (private model repo)
- Update `config.py` in the Space to point right panel at the new checkpoint
- Measure Phase 2 solve rate vs Phase 1 baseline using the existing Space

**Out of scope (Phase 3):**
- DPO / preference pairs from Phase 1 logs
- Iterative fine-tuning (train → collect → retrain)
- Making anything public
- Chess or any second environment
- Benchmarking against other models

---

## 3. New Repo Structure

```
knee-maze/
├── ...existing files unchanged...
├── solver.py              # BFS optimal path finder
├── generate_dataset.py    # runs solver on N mazes, uploads SFT dataset to HF
├── train_modal.py         # Modal app: downloads dataset, SFT, uploads checkpoint
└── specv3.md
```

No changes to `maze.py`, `agent.py`, `runner.py`, `app.py`. Only `config.py` gets one line updated at the end when the checkpoint is ready.

---

## 4. Module Specs

### `solver.py`

```python
def solve_maze(maze: Maze) -> list[str]:
    # BFS from maze.start to maze.end
    # returns list of directions e.g. ["right", "right", "down", "down", ...]
    # empty list = unsolvable (should never happen for a perfect maze)
```

- Uses `collections.deque`, BFS, visits each cell at most once
- Deterministic — same maze always gives same optimal path
- Quick sanity test: `len(solve_maze(generate_maze(8, 42))) > 0`

### `generate_dataset.py`

Generates the SFT training dataset and pushes it to `bolajiev/knee-maze-logs` (same dataset repo, different subfolder).

**What one training example looks like:**

Each step along a BFS-optimal trajectory becomes one chat example:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "You are navigating a text maze. Your position is @, the goal is E.\n\n#################\n#@..#...........#\n...\n#################\n\nValid moves: right, down\n\nReply with exactly one word — up, down, left, or right."
    },
    {
      "role": "assistant",
      "content": "right"
    }
  ]
}
```

The user message is **identical** to what `model_agent` sends during inference. The assistant message is the BFS-optimal direction. This trains the model to reproduce optimal behavior in the exact prompt format it will see at inference time.

**Generation loop:**

```python
for seed in range(N_MAZES):           # N_MAZES = 10_000
    maze = generate_maze(8, seed)
    path = solve_maze(maze)            # BFS optimal directions
    pos = maze.start
    for direction in path:
        grid = render(maze, pos)
        valid_moves = maze.valid_moves(pos)
        # build user message (same prompt as model_agent)
        # build assistant message = direction
        # append to examples list
        pos = step(pos, direction)     # advance position
```

**Output:** ~150,000–200,000 examples (10k mazes × ~15-20 steps avg).

Upload as a single JSONL to HF Dataset:
```
bolajiev/knee-maze-logs
└── sft/train.jsonl
```

CLI:
```
python generate_dataset.py --n-mazes 10000 --maze-size 8
```

### `train_modal.py`

Modal app, runs on a GPU (T4 or A10G). Estimated runtime: 30-60 min.

**Steps:**
1. Download `sft/train.jsonl` from HF Dataset repo
2. Load `Qwen/Qwen2.5-1.5B-Instruct` from HF Hub
3. Apply LoRA via `peft` (r=16, target_modules for Qwen attention layers)
4. Fine-tune with `trl.SFTTrainer`:
   - `max_seq_length=512` (maze prompt fits easily)
   - 3 epochs
   - batch size 8, gradient accumulation 4 → effective batch 32
   - learning rate 2e-4, cosine schedule
5. Push merged checkpoint to HF Hub as `bolajiev/qwen-maze-sft` (private)

**LoRA config:**
```python
LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
```

**Run:**
```
modal run train_modal.py
```

No GPU locally needed. Modal handles the machine provisioning.

---

## 5. Connecting Back to the Space

Once `bolajiev/qwen-maze-sft` is on HF Hub, one line in `config.py`:

```python
FINE_TUNED_MODEL_PATH = "bolajiev/qwen-maze-sft"  # was None
```

Push to Space. The right panel (`Fine-tuned`) loads the checkpoint and becomes live — no other code changes needed. This was designed in from Phase 1.

---

## 6. Dataset Summary

| Dataset | Source | Size | Used for |
|---|---|---|---|
| BFS optimal trajectories | `generate_dataset.py` | ~150k–200k examples | Phase 2 SFT (this phase) |
| Phase 1 model run logs | HF Dataset `logs/` | depends on run count | Phase 3 DPO (next phase) |

The Phase 1 logs record `intended_action` (what the model wanted) vs `action` (what was executed) and `wall_hit: true`. This is exactly the structure needed for DPO preference pairs in Phase 3: `(maze_state, good_action=BFS_optimal, bad_action=model_intended_wall_hit)`.

---

## 7. Tech Stack (additions)

```
trl          # SFTTrainer
peft         # LoRA
modal        # GPU compute (free tier: ~30 GPU-hours/month)
datasets     # HuggingFace datasets library for loading JSONL
```

Add to `requirements.txt` only `datasets` (needed by `generate_dataset.py` locally). `trl`, `peft`, `modal` are only needed in the Modal training environment — they go in the Modal image definition inside `train_modal.py`, not in `requirements.txt`.

---

## 8. Definition of Done (Phase 2 exit condition)

- [ ] `solver.py` BFS finds a solution for every maze, deterministic, length <= 2×(2×size) steps
- [ ] `generate_dataset.py` produces valid JSONL, all examples round-trip through `json.loads()`, uploads to HF Dataset
- [ ] Modal training run completes without OOM, checkpoint pushed to `bolajiev/qwen-maze-sft`
- [ ] Fine-tuned panel in the Space loads and runs the checkpoint
- [ ] Fine-tuned model solve rate on 50 episodes (8x8) is **measurably higher** than Phase 1 baseline
- [ ] Side-by-side comparison screenshot: Base vs Fine-tuned solve rates

**Target:** Phase 1 baseline is likely 5-15%. After SFT on 150k optimal examples, target is >50%. If it's not, investigate before moving to Phase 3.

---

## 9. Phase 3 Preview (do not build yet)

Phase 3 uses the Phase 1 logs as DPO preference pairs:
- **Chosen:** BFS-optimal action for that maze state
- **Rejected:** the model's `intended_action` when `wall_hit: true`

This teaches the model not just what the right answer is, but to move away from its specific failure patterns. DPO on top of the Phase 2 SFT checkpoint, not the base model.
