"""
Research audit eval — 4 agents, guardrail fire rate, BFS-optimal decision rate.

Agents:
  greedy   — rule-based, always picks BFS-optimal move (theoretical ceiling)
  finetuned — our trained Qwen 0.5B
  base      — untrained Qwen 0.5B (same weights, no fine-tuning)
  random    — uniform random valid move + guardrail

Key metrics beyond solve rate:
  guardrail_rate  — % of steps where guardrail overrode the model
  bfs_optimal_rate — % of steps where the executed move reduced BFS distance
  path_efficiency  — optimal_steps / actual_steps (1.0 = perfect)

Usage:
    .venv/bin/modal run eval_research_modal.py
    .venv/bin/modal run eval_research_modal.py --episodes 20 --sizes 7,8,11
"""
import os
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4.0",
        "transformers>=4.45.0,<5.0.0",
        "accelerate>=0.34.0",
        "huggingface_hub>=0.25.0",
    )
    .add_local_dir("/root/KNEE/KNEE-MAZE", remote_path="/root/project", copy=True)
)

app = modal.App("knee-maze-research-eval", image=image)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

BASE_MODEL      = "Qwen/Qwen2.5-0.5B-Instruct"
FINETUNED_MODEL = "bolajiev/qwen-maze-traces"
EPISODES        = 20
SIZES           = [6, 7, 8, 10, 11, 12]


@app.function(
    gpu="T4",
    timeout=7200,
    secrets=[modal.Secret.from_name("hf-token")],
    volumes={"/root/.cache/huggingface": hf_cache},
)
def run_research_eval(episodes: int = EPISODES, sizes: list[int] = SIZES):
    import sys, random, statistics, math
    sys.path.insert(0, "/root/project")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from maze import DIRECTIONS, generate_maze
    from solver import bfs_distance_map, solve_maze
    from agent import make_structured_prompt, _parse_action, _generate

    hf_token = os.environ["HF_TOKEN"]

    # ── Agents ────────────────────────────────────────────────────────────────

    def greedy_agent(state, bfs_map, **_):
        """Rule-based oracle: always picks the valid move that most reduces BFS distance."""
        pos = state["position"]
        best_move, best_dist = None, float("inf")
        for move in state["valid_moves"]:
            dr, dc = DIRECTIONS[move]
            d = bfs_map.get((pos[0] + dr, pos[1] + dc), 9999)
            if d < best_dist:
                best_dist, best_move = d, move
        return best_move or random.choice(state["valid_moves"])

    def random_agent(state, **_):
        return random.choice(state["valid_moves"])

    def make_model_agent(mdl, tok):
        def agent(state, **_):
            prompt = make_structured_prompt(state)
            try:
                raw = _generate(prompt, mdl, tok, max_new_tokens=8)
                action = _parse_action(raw)
                if action and action in state["valid_moves"]:
                    return action
            except Exception:
                pass
            # fallback
            try:
                fb = f"Choose one: {', '.join(state['valid_moves'])}. Reply with that word only."
                raw2 = _generate(fb, mdl, tok, max_new_tokens=8)
                action = _parse_action(raw2)
                if action and action in state["valid_moves"]:
                    return action
            except Exception:
                pass
            return random.choice(state["valid_moves"])
        return agent

    # ── Instrumented episode runner ────────────────────────────────────────────

    def run_episode(agent_fn, maze_size, seed):
        """Returns detailed episode stats including guardrail fires and BFS-optimal rate."""
        maze = generate_maze(maze_size, seed)
        bfs_map = bfs_distance_map(maze)
        pos = maze.start
        history = []
        guardrail_fires = 0
        bfs_optimal_steps = 0
        total_steps = 0

        for step in range(1, 201):
            valid_moves = maze.valid_moves(pos)
            r, c = pos
            walls = {d: not maze.can_move(pos, (r + dr, c + dc))
                     for d, (dr, dc) in DIRECTIONS.items()}
            state = {
                "position": pos, "valid_moves": valid_moves, "walls": walls,
                "bfs_dist": bfs_map.get(pos, -1), "goal": maze.end,
                "maze_size": maze_size, "history": list(history[-10:]),
            }

            action = agent_fn(state, bfs_map=bfs_map)

            # Guardrail: random non-revisiting override (not BFS-guided)
            recent = set(history[-6:])
            dr2, dc2 = DIRECTIONS[action]
            would_revisit = (pos[0] + dr2, pos[1] + dc2) in recent
            if would_revisit:
                alts = [m for m in valid_moves
                        if (pos[0] + DIRECTIONS[m][0], pos[1] + DIRECTIONS[m][1]) not in recent]
                if alts:
                    action = random.choice(alts)
                    guardrail_fires += 1

            # Was the executed move BFS-optimal?
            current_dist = bfs_map.get(pos, 0)
            dr3, dc3 = DIRECTIONS[action]
            next_dist = bfs_map.get((pos[0] + dr3, pos[1] + dc3), current_dist + 1)
            if next_dist < current_dist:
                bfs_optimal_steps += 1
            total_steps += 1

            history.append(pos)
            pos = (pos[0] + dr3, pos[1] + dc3)

            if pos == maze.end:
                return {
                    "solved": True, "steps": step,
                    "guardrail_rate": guardrail_fires / step,
                    "bfs_optimal_rate": bfs_optimal_steps / total_steps,
                }

        return {
            "solved": False, "steps": 200,
            "guardrail_rate": guardrail_fires / 200,
            "bfs_optimal_rate": bfs_optimal_steps / total_steps,
        }

    # ── Eval one agent across sizes ────────────────────────────────────────────

    def eval_agent(label, agent_fn, sizes, episodes, optimal_steps):
        print(f"\n--- {label} ---")
        results = {}
        seeds = [random.randint(0, 2**31 - 1) for _ in range(episodes * max(sizes))]
        seed_idx = 0

        for size in sizes:
            ep_results = []
            for ep in range(episodes):
                seed = seeds[seed_idx % len(seeds)]
                seed_idx += 1
                ep_results.append(run_episode(agent_fn, size, seed))

            wins = [r for r in ep_results if r["solved"]]
            n = len(ep_results)
            solve_pct = len(wins) / n * 100
            avg_steps = statistics.mean(r["steps"] for r in wins) if wins else 0
            avg_guardrail = statistics.mean(r["guardrail_rate"] for r in ep_results) * 100
            avg_bfs_opt = statistics.mean(r["bfs_optimal_rate"] for r in ep_results) * 100
            efficiency = (optimal_steps[size] / avg_steps * 100) if avg_steps > 0 else 0

            # 95% confidence interval on solve rate
            p = solve_pct / 100
            margin = 1.96 * math.sqrt(p * (1 - p) / n) * 100 if 0 < p < 1 else 0

            results[size] = {
                "solve_pct": solve_pct, "margin": margin,
                "avg_steps": avg_steps, "efficiency": efficiency,
                "guardrail_pct": avg_guardrail, "bfs_optimal_pct": avg_bfs_opt,
                "wins": len(wins), "n": n,
            }
            print(f"  {size}×{size}: {len(wins)}/{n} solved "
                  f"| steps {avg_steps:.1f} ({efficiency:.0f}% eff) "
                  f"| guardrail {avg_guardrail:.1f}% "
                  f"| BFS-opt {avg_bfs_opt:.1f}%")
        return results

    # ── Compute optimal path lengths ──────────────────────────────────────────

    print("Computing optimal path lengths...")
    optimal_steps = {}
    for size in sizes:
        lengths = []
        for s in range(200):
            m = generate_maze(size, s)
            p = solve_maze(m)
            if p:
                lengths.append(len(p))
        optimal_steps[size] = statistics.mean(lengths)
        print(f"  {size}×{size}: mean optimal = {optimal_steps[size]:.1f} steps")

    # ── Load models ───────────────────────────────────────────────────────────

    def load_model(model_id):
        print(f"Loading {model_id}...")
        tok = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        mdl = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="auto", token=hf_token,
        )
        mdl.eval()
        return mdl, tok

    # ── Run all agents ────────────────────────────────────────────────────────

    print(f"\n{'='*70}")
    print(f"Research Eval: {episodes} episodes × {len(sizes)} sizes")
    print(f"Agents: greedy oracle / fine-tuned / base / random")
    print(f"{'='*70}")

    greedy_results = eval_agent("Greedy Oracle (ceiling)", greedy_agent, sizes, episodes, optimal_steps)

    ft_m, ft_t = load_model(FINETUNED_MODEL)
    ft_agent = make_model_agent(ft_m, ft_t)
    ft_results = eval_agent("Fine-tuned (qwen-maze-traces)", ft_agent, sizes, episodes, optimal_steps)
    del ft_m, ft_t

    base_m, base_t = load_model(BASE_MODEL)
    base_agent = make_model_agent(base_m, base_t)
    base_results = eval_agent("Base (Qwen 0.5B)", base_agent, sizes, episodes, optimal_steps)
    del base_m, base_t

    random_results = eval_agent("Random + guardrail", random_agent, sizes, episodes, optimal_steps)

    # ── Print research table ──────────────────────────────────────────────────

    print(f"\n{'='*90}")
    print(f"RESEARCH RESULTS  ({episodes} eps/size, 95% CI on solve rate)")
    print(f"{'='*90}")
    print(f"{'Size':<5} {'Agent':<22} {'Solve':>10} {'Steps':>7} {'Eff%':>6} {'Guardrail%':>11} {'BFS-opt%':>9}")
    print(f"{'-'*90}")

    for size in sizes:
        opt = optimal_steps[size]
        for label, res in [
            ("Greedy oracle", greedy_results),
            ("Fine-tuned", ft_results),
            ("Base", base_results),
            ("Random", random_results),
        ]:
            r = res[size]
            solve_str = f"{r['solve_pct']:.0f}%±{r['margin']:.0f}%"
            print(f"{size}×{size:<3} {label:<22} {solve_str:>10} "
                  f"{r['avg_steps']:>7.1f} {r['efficiency']:>5.0f}% "
                  f"{r['guardrail_pct']:>10.1f}% {r['bfs_optimal_pct']:>8.1f}%")
        print(f"      {'optimal':22} {'—':>10} {opt:>7.1f} {'100%':>6}")
        print(f"{'-'*90}")

    print(f"\nKEY:")
    print(f"  Eff%        = optimal_steps / actual_steps × 100 (higher = better)")
    print(f"  Guardrail%  = % of steps where anti-oscillation override fired")
    print(f"  BFS-opt%    = % of steps where executed move reduced BFS distance")
    print(f"\nIf FT BFS-opt% > Base BFS-opt% → model genuinely learned to follow oracle")
    print(f"If FT guardrail% << Base guardrail% → model loops less, guardrail is a safety net")
    print(f"If FT ≈ greedy oracle → perfect oracle use, training worked")


@app.local_entrypoint()
def main(episodes: int = EPISODES, sizes: str = ",".join(map(str, SIZES))):
    size_list = [int(s) for s in sizes.split(",")]
    run_research_eval.remote(episodes=episodes, sizes=size_list)
