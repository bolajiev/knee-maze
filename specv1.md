# knee-maze — Phase 1 Spec

**Project:** `knee-maze`
**Phase:** 1 — Baseline loop (maze + agent + logging, no training)
**Goal:** Prove the environment + logging pipeline works end-to-end and produce a baseline solve-rate number for Qwen 2.5 on text mazes.

---

## 1. Scope

**In scope (build this):**
- Procedural solvable maze generator + text renderer
- Two interchangeable agents: random-move and Qwen-via-OpenRouter
- Per-step and per-episode JSONL logging
- A runner script that plays N episodes and prints a solve-rate report

**Out of scope (do NOT build yet — later phases):**
- Fine-tuning / Modal / GPU training
- Cloudflare R2 / FastAPI / any backend or server
- DPO, preference pairs, win/loss pairing
- Chess or any second environment
- Benchmarking against GPT-4o / Claude / other models
- HuggingFace dataset or model release

If you find yourself building any of the above, stop — it's not Phase 1.

---

## 2. Tech Stack

- **Language:** Python 3.11+
- **Model access:** OpenRouter API, model `qwen/qwen-2.5-7b-instruct` (configurable string)
- **Storage:** local JSONL files only. No database, no cloud.
- **Dependencies:** `requests` (or `openai` SDK pointed at OpenRouter's base URL), `python-dotenv`. Keep dependencies minimal.

---

## 3. Repo Structure

```
knee-maze/
├── maze.py            # maze generation + rendering
├── agent.py           # random_agent + qwen_agent
├── logger.py           # JSONL writer
├── run_episodes.py    # main runner / CLI entrypoint
├── config.py           # constants + arg parsing helpers
├── .env.example        # OPENROUTER_API_KEY=
├── requirements.txt
├── logs/                # gitignored, output JSONL lands here
└── README.md
```

---

## 4. Module Specs

### `maze.py`
- `generate_maze(size: int, seed: int) -> Maze` — produces a **solvable** maze using a randomized recursive-backtracker (perfect maze: exactly one path between any two cells). Deterministic given the same seed.
- `Maze` should expose: grid dimensions, wall lookup (can I move from cell A to cell B), start position, end position.
- `render(maze: Maze, agent_pos: tuple) -> str` — returns a text grid using:
  - `#` = wall
  - `.` = open path
  - `S` = start
  - `E` = end (goal)
  - `@` = current agent position (drawn over whatever's underneath)
- Maze size should be configurable (default: 8x8 cells).

### `agent.py`
Single interface both agents implement:
```python
def get_action(state: dict) -> dict:
    # returns {"action": "up"|"down"|"left"|"right", "raw_output": str|None}
```
- `state` passed in should contain: rendered text grid, current position, list of currently-valid moves (for the agent to optionally use, not required).
- **`random_agent`** — picks uniformly from the valid moves list. Build and test this first to validate the full pipeline before any API calls.
- **`qwen_agent`** — builds a prompt containing the rendered maze + a strict instruction to respond with exactly one word (`up`/`down`/`left`/`right`), calls OpenRouter, parses the response.
  - If the response doesn't parse cleanly: retry once with a stricter prompt, then fall back to a random valid move. Log this fallback (`raw_output` preserved, `valid_move` still computed normally).

### `logger.py`
- Append-only JSONL writer. One file per run: `logs/run_<timestamp>_<agent_type>.jsonl`
- Two record types, both written to the same file, distinguished by a `"record_type"` field.

**Step record:**
```json
{
  "record_type": "step",
  "run_id": "string",
  "episode_id": 0,
  "step": 0,
  "maze_seed": 12345,
  "agent_type": "random",
  "position_before": [0, 0],
  "action": "right",
  "valid_move": true,
  "position_after": [0, 1],
  "status": "ongoing",
  "raw_model_output": null,
  "timestamp": "ISO8601"
}
```
`status` is one of: `ongoing`, `win`, `wall_hit`, `timeout`.

**Episode summary record:**
```json
{
  "record_type": "episode_summary",
  "run_id": "string",
  "episode_id": 0,
  "maze_seed": 12345,
  "agent_type": "random",
  "maze_size": 8,
  "solved": true,
  "total_steps": 14,
  "wall_hits": 2,
  "reason_ended": "win"
}
```

### `run_episodes.py`
CLI entrypoint:
```
python run_episodes.py --agent random --episodes 100 --maze-size 8 --max-steps 200
python run_episodes.py --agent qwen   --episodes 100 --maze-size 8 --max-steps 200 \
    --model qwen/qwen-2.5-7b-instruct --temperature 0.2
```
Loop, per episode:
1. Generate a fresh maze (new seed each episode, but log the seed for reproducibility)
2. Step until: win, wall-hit-triggers-loss (decide: wall hit = wasted step, not episode end, unless you want strict mode — default: wall hit just doesn't move, episode continues), or `max_steps` reached → timeout
3. Log every step + the episode summary at the end

At the end of all episodes, print a report to console:
- Solve rate (% of episodes won)
- Average steps-to-solve (solved episodes only)
- Average wall-hit count per episode
- Timeout rate

### `config.py`
- Reads `OPENROUTER_API_KEY` from `.env`
- Holds defaults: maze size, max steps, model name, temperature

---

## 5. Definition of Done (Phase 1 exit condition)

- [ ] `random_agent` completes a 50-episode run with zero crashes, produces valid JSONL
- [ ] `qwen_agent` completes a 50-episode run against the live OpenRouter API with <5% unparseable/fallback responses
- [ ] Console report prints solve rate, avg steps, wall-hit rate, timeout rate — separately for each agent type
- [ ] JSONL logs are loadable line-by-line with `json.loads()` with no malformed rows
- [ ] Maze generation is deterministic given a seed (same seed → same maze, verified by a quick test)

Once this is done and you have a baseline solve-rate number for Qwen 2.5 7B on, say, 8x8 mazes — **stop**. That number is the input to Phase 2 (fine-tuning), which is a separate spec.
