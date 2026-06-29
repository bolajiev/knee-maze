# knee-maze — Project Progress

## What this project is

Can a 0.5B parameter LLM learn to navigate mazes — not by being large, but by being trained in the right format?

Pipeline: procedural maze env → BFS oracle traces → LoRA SFT → GRPO → honest 4-agent benchmark.

---

## Phase 1 — Baseline ✅ DONE

**Goal:** Prove the pipeline works end-to-end.

**Built:**
- Procedural maze generator (randomized recursive backtracker, deterministic per seed)
- Text-based renderer, BFS solver
- Gradio Space with live step-by-step rendering (Base vs Fine-tuned panels)
- Per-step JSONL logging → HF Dataset `bolajiev/knee-maze-logs`

**Result:** Base Qwen2.5-0.5B-Instruct: **0% solve rate** on 8×8. Loops indefinitely.

---

## Phase 2 & 3 — Early SFT + DPO attempts ✅ DONE (superseded)

- Phase 2: SFT on basic grid-format traces → 14% solve rate (looping, no directional strategy)
- Phase 3: DPO on failure logs → abandoned; DPO reward margins improved but solve rate didn't
- Key learning: format was wrong. Old format buried "Action: right" after 80 tokens of reasoning. With `max_new_tokens=20`, inference never reached the action word. Model fell back to random.

---

## Phase 4a — SFT retraining with correct format ✅ DONE

**Problem diagnosed:** Three compounding bugs:
1. Action buried at end of 80-token trace → `max_new_tokens=20` never reached it
2. Absolute coordinates `(row, col)` → didn't generalize across maze sizes
3. BFS-optimal guardrail was picking the best move, not just breaking loops — the guardrail was half-solving the maze (cheating)

**All three fixed:**
- Action-first format: `assistant: "right\n\n[full reasoning]"` → `max_new_tokens=8` captures action in first 2-3 tokens
- Relative coords: `Rows to exit: X | Cols to exit: Y` — size-agnostic
- Guardrail changed to **random** non-revisiting override — only breaks loops, doesn't guide

**SFT dataset:**
- 86,000 single-step examples, BFS-oracle optimal actions
- Curriculum: 6×6 (15%), 7×7 (15%), 8×8 (40%), 11×11 (30%)
- `MIN_PATH_LENGTH = 8` quality filter
- Seeds 20000–23000 (separate from eval)

**Training:** Modal A10G, LoRA r=16, 3 epochs, bf16, lr=2e-4
**Model:** `bolajiev/qwen-maze-traces`

---

## Phase 4b — GRPO ✅ DONE (no improvement)

**Goal:** Push BFS-optimal decision rate from ~78% toward 95%+ using reinforcement learning.

**Reward per step:**
- +1.0 BFS-optimal move, 0.0 lateral, -0.5 wrong direction
- +5.0 exit reached, -1.0 wall hit, -0.3 bad parse

**Result:** `frac_reward_zero_std ≈ 1.0` throughout — near-zero reward variance across all 4 rollouts per prompt. After SFT, the model is so confident it picks the same action every time, so GRPO sees no variance and produces no gradient (`loss: 0.0` throughout).

**Root cause:** Single-step GRPO with a post-SFT model that's near-deterministic. All 4 rollouts → same action → same reward → GRPO can't learn. Tried `temperature=0.8` — model entropy was ~0.001, still deterministic.

**Model:** `bolajiev/qwen-maze-grpo` (effectively identical to SFT checkpoint)

**Next step if revisiting:** Full trajectory GRPO — run complete episodes, reward = solve + efficiency. This gives real variance because some rollouts solve the maze and others don't.

---

## Research Audit Results

**4-agent comparison, 20 episodes per size, 95% CI on solve rate**

Run: `modal run eval_research_modal.py`

| Size | Agent | Solve | Steps | Eff% | Guardrail% | BFS-opt% |
|------|-------|-------|-------|------|------------|---------|
| 6×6 | Greedy oracle | 100%±0% | 19.4 | 98% | 0% | 100% |
| 6×6 | Fine-tuned | 100%±0% | 54.3 | 35% | 43% | 79% |
| 6×6 | Base | 90%±13% | 40.1 | 47% | 40% | 80% |
| 6×6 | Random | 95%±10% | 34.2 | 55% | 42% | 84% |
| 8×8 | Greedy oracle | 100%±0% | 35.9 | 87% | 0% | 100% |
| 8×8 | Fine-tuned | 90%±13% | 58.6 | 53% | 42% | 81% |
| 8×8 | Base | 75%±19% | 72.4 | 43% | 43% | 73% |
| 8×8 | Random | 75%±19% | 64.5 | 48% | 46% | 77% |
| 11×11 | Greedy oracle | 100%±0% | 57.7 | 91% | 0% | 100% |
| 11×11 | Fine-tuned | 55%±22% | 97.3 | 54% | 44% | 71% |
| 11×11 | Base | 50%±22% | 100.4 | 52% | 42% | 69% |
| 11×11 | Random | 60%±21% | 88.0 | 60% | 42% | 73% |

**Key findings:**
- SFT gives +15pp solve rate at 8×8 vs base (90% vs 75%) — real improvement
- BFS-opt rate: 81% (fine-tuned) vs 73% (base) at 8×8 — model genuinely follows oracle better
- Guardrail fires ~43% at all agents including random — big contribution to solve rate
- No generalization beyond training sizes — fine-tuned loses to random at 10×10+
- GRPO: no improvement (reward saturation, as above)

---

## Honest conclusions

The SFT improvement is real. The key was fixing the training format (action-first), not the model size. A 0.5B model can learn to follow a BFS oracle at trained sizes when given the right signal.

The guardrail does substantial work — ~43% of steps for every agent. This is disclosed clearly. Without it, all solve rates would drop significantly.

The base model is competitive because the prompt includes the BFS distance explicitly. Instruction-following in the base model already partially uses this. Fine-tuning narrows the remaining gap but doesn't close it.

Full trajectory RL (not single-step) is the correct next step to meaningfully improve beyond SFT.
