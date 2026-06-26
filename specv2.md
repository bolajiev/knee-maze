# knee-maze — Phase 1 Spec (v2: HF Space architecture)

**Project:** `knee-maze`
**Phase:** 1 — Baseline loop (maze + agent + live viz + logging, no training)

**What changed from v1:** No external API (OpenRouter/Fireworks). Qwen2.5-1.5B-Instruct now loads directly inside a private Hugging Face Space via `transformers` — no rate limits, no API key for inference. Logs go to a private HF Dataset repo instead of local JSONL files (disk on Spaces is ephemeral). UI is a live Gradio app with two panels: Base model (active now) and Fine-tuned model (placeholder until Phase 2).

---

## 1. Scope

**In scope:**
- Procedural solvable maze generator + text renderer
- Two agents: random-move, and Qwen2.5-1.5B-Instruct loaded in-process
- Live Gradio UI, two panels (Base / Fine-tuned), step-by-step maze rendering
- Step + episode logging, pushed to a private HF Dataset repo
- Shared core loop usable both headless (for fast local testing) and from the Space UI

**Out of scope (later phases):**
- Fine-tuning / Modal / GPU training
- DPO, preference pairs
- Chess or any second environment
- Benchmarking against other models
- Public visibility (Space stays **private** in Phase 1)

---

## 2. Architecture

```
HF Space (private, Gradio SDK, CPU Basic — 2 vCPU / 16GB RAM, free)
│
├── app.py            → Gradio UI, two panels, live step rendering
├── model_loader.py    → loads + caches Qwen2.5-1.5B-Instruct via transformers
├── maze.py            → maze gen + text rendering
├── agent.py           → random_agent + model_agent (shared interface)
├── runner.py           → core episode loop (used by app.py AND a headless test script)
├── logger.py           → builds step/episode records, batches, pushes to HF Dataset
└── requirements.txt
```

No OpenRouter, no Fireworks, no local file persistence assumed — disk wipes on Space restart/sleep, so logs must land in the Dataset repo, not just `logs/`.

---

## 3. Repo / Space Structure

```
knee-maze/
├── app.py
├── model_loader.py
├── maze.py
├── agent.py
├── runner.py
├── logger.py
├── config.py
├── test_headless.py     # quick local sanity check, no Gradio needed
├── requirements.txt
└── README.md             # must include HF Space config frontmatter (see §6)
```

---

## 4. Module Specs

### `maze.py`
Unchanged from v1 logic:
- `generate_maze(size: int, seed: int) -> Maze` — randomized recursive-backtracker, deterministic per seed, guaranteed solvable.
- `render(maze, agent_pos) -> str` — text grid: `#` wall, `.` open, `S` start, `E` end, `@` agent.

### `model_loader.py`
- `load_model(model_path: str)` — loads tokenizer + model via `transformers.AutoModelForCausalLM` / `AutoTokenizer`, `torch_dtype="auto"`, CPU device map. Cache the loaded object (module-level singleton or `functools.lru_cache`) so it only loads once per Space session, not per request.
- Two call sites: `load_model("Qwen/Qwen2.5-1.5B-Instruct")` for the Base panel. The Fine-tuned panel calls this with a placeholder path that doesn't exist yet — `app.py` should catch this and show "not trained yet" instead of crashing.

### `agent.py`
Same interface as v1:
```python
def get_action(state: dict, model=None, tokenizer=None) -> dict:
    # returns {"action": "up"|"down"|"left"|"right", "raw_output": str|None}
```
- `random_agent` — uniform pick from valid moves. Build and test first.
- `model_agent` — builds prompt (rendered grid + strict "respond with exactly one word" instruction) via `tokenizer.apply_chat_template`, calls `model.generate(max_new_tokens=8, do_sample=False)`, parses output. Parse failure → retry once, then fall back to random valid move (log the fallback).

### `runner.py`
Core loop, **framework-agnostic** (no Gradio imports here) so it's reusable headless:
```python
def run_episode(agent_fn, maze_size, max_steps, seed) -> dict:
    # generates maze, steps until win/timeout, returns full episode record
    # yields (or returns a list of) per-step snapshots for live rendering
```
Make this a generator (`yield` each step's render + record) so `app.py` can stream it into the UI, and `test_headless.py` can just exhaust it in a loop.

### `logger.py`
- Accumulates step + episode records in memory during a run.
- `flush_to_dataset(records: list, run_id: str)` — writes records to a JSONL file, uploads via `huggingface_hub.HfApi().upload_file(path_in_repo=f"logs/{run_id}.jsonl", repo_id=<your-dataset-repo>, repo_type="dataset")`.
- Call this **once per episode** (not per step) to avoid hammering the Hub API. A run of 50 episodes = 50 small upload calls, which is fine.
- Needs `HF_TOKEN` (write-access) set as a **Space secret** — never hardcode it.

Record schemas are unchanged from v1 (`record_type: "step"` / `"episode_summary"`, same fields).

### `app.py`
Gradio Blocks app, two columns:

**Left column — "Base (Qwen2.5-1.5B)"**
- Controls: episode count, max steps, "Run" button
- On click: streams `runner.run_episode(...)` output, re-rendering the maze text grid after each step (small `time.sleep(0.1–0.3)` between steps for visibility)
- After all episodes: shows solve rate / avg steps / wall-hit rate summary, calls `logger.flush_to_dataset(...)`

**Right column — "Fine-tuned"**
- Same UI shell, but `model_loader.load_model(FINE_TUNED_PATH)` is wrapped in try/except
- If the model doesn't exist yet: show a static message — *"No fine-tuned model yet. Run Phase 2 first."* — button disabled
- Once Phase 2 produces a checkpoint, this panel goes live with zero code changes beyond the path

### `config.py`
- `BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"`
- `FINE_TUNED_MODEL_PATH = None  # set after Phase 2`
- `DATASET_REPO_ID = "<your-username>/knee-maze-logs"` (private dataset repo)
- Defaults: maze size, max steps

---

## 5. HF Space Config

`README.md` must start with this frontmatter block for the Space to build correctly:
```yaml
---
title: knee-maze
emoji: 🧭
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: "<current gradio version>"
app_file: app.py
pinned: false
private: true
---
```

`requirements.txt`:
```
gradio
transformers
torch
huggingface_hub
accelerate
```

**Secrets to set in Space settings:** `HF_TOKEN` (write access, for dataset push only — not needed for downloading the public Qwen model).

---

## 6. Definition of Done (Phase 1 exit condition)

- [ ] Space builds and runs privately on CPU Basic hardware
- [ ] `test_headless.py` runs 20 episodes with `random_agent` end-to-end with no Gradio dependency, no crashes
- [ ] Base panel runs `model_agent` (Qwen2.5-1.5B) live in the UI, maze grid visibly updates step by step
- [ ] Fine-tuned panel renders its placeholder state without crashing the app
- [ ] After a run, JSONL logs are visible in the private HF Dataset repo
- [ ] Console/UI report shows solve rate, avg steps-to-solve, wall-hit rate, timeout rate for the Base model run

Once you have a baseline solve-rate number for Qwen2.5-1.5B on 8x8 mazes from this Space — **stop**. That number feeds Phase 2 (fine-tuning), which is a separate spec.
