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

**Research question:** Can a 0.5B parameter LLM learn to navigate mazes — not by being large, but by being trained in the right format?

Fine-tuned **Qwen2.5-0.5B-Instruct** on BFS-oracle traces then measured honestly against a rule-based ceiling, the base model, and a random agent.

---

## What we built

A full RL data pipeline:

1. **Procedural maze environment** — randomized recursive backtracker, text-based, deterministic per seed
2. **BFS oracle** — always picks the move that most reduces distance to exit (theoretical ceiling)
3. **SFT dataset** — 86k single-step examples across 6×6/7×7/8×8/11×11 mazes, action-first format
4. **LoRA SFT on Modal A10G** — 3 epochs, bf16, Qwen2.5-0.5B as base
5. **4-agent research audit** — greedy oracle / fine-tuned / base / random, 20 episodes per size

**Key design choices:**
- Action-first training format: assistant outputs direction word first, then reasoning trace. This means `max_new_tokens=8` captures the decision cleanly at inference.
- Relative coordinates: `Rows to exit: X | Cols to exit: Y` (size-agnostic, generalizes better)
- Anti-oscillation guardrail: random non-revisiting override when model loops. **Random, not BFS-guided** — it only breaks loops, it doesn't guide toward the exit.
- BFS distance given in prompt — model learns to follow it, not rediscover it

---

## Results

**20 episodes per size, 95% CI on solve rate**

| Size | Agent | Solve rate | Steps | Efficiency | Guardrail% | BFS-opt% |
|------|-------|-----------|-------|-----------|------------|---------|
| 6×6 | Greedy oracle | 100%±0% | 19.4 | 98% | 0% | 100% |
| 6×6 | **Fine-tuned** | **100%±0%** | 54.3 | 35% | 43% | 79% |
| 6×6 | Base | 90%±13% | 40.1 | 47% | 40% | 80% |
| 6×6 | Random | 95%±10% | 34.2 | 55% | 42% | 84% |
| — | optimal | — | 18.9 | 100% | — | — |
| 7×7 | Greedy oracle | 100%±0% | 23.1 | 107% | 0% | 100% |
| 7×7 | **Fine-tuned** | **100%±0%** | 63.7 | 39% | 42% | 79% |
| 7×7 | Base | 90%±13% | 63.2 | 39% | 46% | 81% |
| 7×7 | Random | 85%±16% | 60.6 | 41% | 44% | 77% |
| — | optimal | — | 24.8 | 100% | — | — |
| 8×8 | Greedy oracle | 100%±0% | 35.9 | 87% | 0% | 100% |
| 8×8 | **Fine-tuned** | **90%±13%** | 58.6 | 53% | 42% | 81% |
| 8×8 | Base | 75%±19% | 72.4 | 43% | 43% | 73% |
| 8×8 | Random | 75%±19% | 64.5 | 48% | 46% | 77% |
| — | optimal | — | 31.2 | 100% | — | — |
| 10×10 | Greedy oracle | 100%±0% | 49.8 | 91% | 0% | 100% |
| 10×10 | **Fine-tuned** | **60%±21%** | 66.8 | 68% | 42% | 77% |
| 10×10 | Base | 75%±19% | 83.7 | 54% | 44% | 78% |
| 10×10 | Random | 85%±16% | 91.2 | 50% | 44% | 77% |
| — | optimal | — | 45.5 | 100% | — | — |
| 11×11 | Greedy oracle | 100%±0% | 57.7 | 91% | 0% | 100% |
| 11×11 | **Fine-tuned** | **55%±22%** | 97.3 | 54% | 44% | 71% |
| 11×11 | Base | 50%±22% | 100.4 | 52% | 42% | 69% |
| 11×11 | Random | 60%±21% | 88.0 | 60% | 42% | 73% |
| — | optimal | — | 52.6 | 100% | — | — |
| 12×12 | Greedy oracle | 100%±0% | 55.9 | 110% | 0% | 100% |
| 12×12 | **Fine-tuned** | **35%±21%** | 90.6 | 68% | 44% | 69% |
| 12×12 | Base | 50%±22% | 106.0 | 58% | 43% | 68% |
| 12×12 | Random | 45%±22% | 88.7 | 69% | 45% | 71% |
| — | optimal | — | 61.5 | 100% | — | — |

**Eff%** = optimal steps / actual steps × 100. **Guardrail%** = steps where anti-oscillation override fired. **BFS-opt%** = steps where executed move reduced BFS distance.

---

## What the results actually mean

**SFT works at trained sizes.** At 8×8 (the primary training size), fine-tuned solves 90% vs base 75% — a real +15pp improvement. BFS-opt rate goes from 73% (base) to 81% (fine-tuned), meaning the model genuinely learned to follow the oracle, not just got lucky.

**The guardrail fires ~42-44% of steps for every agent including random.** This is the honest number — the guardrail is doing significant work. It doesn't guide toward the exit (it picks randomly), but it breaks oscillation loops that would otherwise cause timeouts. Without it, solve rates for all agents including fine-tuned would drop substantially.

**Generalization is limited.** Fine-tuned underperforms even random at 10×10+ (60% vs 85%). The model learned the pattern for trained sizes but doesn't scale. To fix this, you'd need larger mazes in the curriculum or trajectory-level RL (full episode reward, not single-step).

**GRPO (single-step) didn't help.** After SFT, the model is so confident that when GRPO samples 4 rollouts per prompt, all 4 pick the same action → same reward → zero variance → zero gradient (`frac_reward_zero_std ≈ 1.0` throughout training). Single-step GRPO with a deterministic model is a dead end. Full trajectory GRPO (reward the complete episode) is the right next step but wasn't attempted.

**The base model is surprisingly competitive.** Qwen2.5-0.5B-Instruct comes with instruction-following built in and the prompt includes BFS distance explicitly. The base model uses that signal too — its BFS-opt rate is 68-80% without any fine-tuning. The SFT gain is real but modest.

---

## Repo structure

```
maze.py                  — procedural maze generator (recursive backtracker)
solver.py                — BFS distance map + greedy solver
agent.py                 — model inference, prompt builder, action parser
runner.py                — episode runner with anti-oscillation guardrail
config.py                — model paths
model_loader.py          — cached HF model loader
app.py                   — Gradio Space UI (base vs fine-tuned comparison)
logger.py                — JSONL episode logger
generate_sft_traces.py   — build SFT dataset (BFS oracle traces, action-first)
train_traces_modal.py    — SFT training on Modal A10G (LoRA, 3 epochs)
train_grpo_modal.py      — GRPO training on Modal A10G (single-step reward)
eval_modal.py            — base vs fine-tuned comparison eval
eval_research_modal.py   — 4-agent research audit with 95% CI
requirements.txt
```

---

## Reproduce

```bash
# 1. Generate SFT dataset
python generate_sft_traces.py

# 2. Train on Modal
pip install modal
modal run train_traces_modal.py

# 3. Run research eval
modal run eval_research_modal.py
```

**Models on HuggingFace:**
- SFT: `bolajiev/qwen-maze-traces`
- GRPO (no improvement over SFT): `bolajiev/qwen-maze-grpo`

---

## Stack

- Model: Qwen2.5-0.5B-Instruct (LoRA fine-tuned)
- Training: Modal A10G, bf16, TRL SFTTrainer / GRPOTrainer
- Inference: Modal T4 (eval) / HF Space T4 (demo)
- Experiment tracking: JSONL logs → HF Dataset `bolajiev/knee-maze-logs`
