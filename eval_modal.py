"""
Eval: run base vs fine-tuned across maze sizes on Modal GPU.
No UI sleep — pure inference. Results printed as a table.

Usage:
    .venv/bin/modal run eval_modal.py
    .venv/bin/modal run eval_modal.py --episodes 20 --sizes 6,7,8,10,11,12
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
    .add_local_dir(
        "/root/KNEE/KNEE-MAZE",
        remote_path="/root/project",
        copy=True,
    )
)

app = modal.App("knee-maze-eval", image=image)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

BASE_MODEL       = "Qwen/Qwen2.5-0.5B-Instruct"
FINETUNED_MODEL  = "bolajiev/qwen-maze-traces"
EPISODES         = 20
SIZES            = [6, 7, 8, 10, 11, 12]


@app.function(
    gpu="T4",
    timeout=3600,
    secrets=[modal.Secret.from_name("hf-token")],
    volumes={"/root/.cache/huggingface": hf_cache},
)
def run_eval(episodes: int = EPISODES, sizes: list[int] = SIZES):
    import sys
    import random
    import statistics
    sys.path.insert(0, "/root/project")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from runner import run_episode
    from agent import model_agent

    hf_token = os.environ["HF_TOKEN"]

    def load(model_id):
        print(f"  Loading {model_id}...")
        tok = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        mdl = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            token=hf_token,
        )
        mdl.eval()
        return mdl, tok

    def eval_model(label, mdl, tok, sizes, episodes):
        results = {}
        for size in sizes:
            wins, steps_wins, wall_hits_all, timeouts = 0, [], [], 0
            for ep in range(episodes):
                seed = random.randint(0, 2**31 - 1)
                summary = None
                for snap in run_episode(
                    model_agent, size, 200, seed,
                    episode_id=ep, model=mdl, tokenizer=tok
                ):
                    if snap["done"]:
                        summary = snap["episode_summary"]
                if summary["solved"]:
                    wins += 1
                    steps_wins.append(summary["total_steps"])
                else:
                    timeouts += 1
                wall_hits_all.append(summary["wall_hits"])
            results[size] = {
                "solve_pct": wins / episodes * 100,
                "wins": wins,
                "avg_steps": statistics.mean(steps_wins) if steps_wins else 0,
                "avg_walls": statistics.mean(wall_hits_all),
                "timeouts": timeouts,
            }
            print(f"  {label} | {size}×{size}: {wins}/{episodes} solved | "
                  f"avg {results[size]['avg_steps']:.1f} steps | "
                  f"avg {results[size]['avg_walls']:.1f} wall hits")
        return results

    # Load BFS optimal for reference
    from maze import generate_maze
    from solver import solve_maze
    optimal = {}
    for size in sizes:
        lengths = [len(solve_maze(generate_maze(size, s))) for s in range(100)
                   if solve_maze(generate_maze(size, s))]
        optimal[size] = statistics.mean(lengths)

    print(f"\n{'='*60}")
    print(f"Eval: {episodes} episodes per size, sizes {sizes}")
    print(f"{'='*60}\n")

    print("--- Base model ---")
    base_m, base_t = load(BASE_MODEL)
    base_results = eval_model("Base      ", base_m, base_t, sizes, episodes)
    del base_m, base_t

    print("\n--- Fine-tuned model ---")
    ft_m, ft_t = load(FINETUNED_MODEL)
    ft_results = eval_model("Fine-tuned", ft_m, ft_t, sizes, episodes)
    del ft_m, ft_t

    # Print comparison table
    print(f"\n{'='*60}")
    print(f"RESULTS  ({episodes} episodes per size)")
    print(f"{'='*60}")
    print(f"{'Size':<6} {'Optimal':>7} | {'Base':>6} {'Steps':>7} | {'FT':>6} {'Steps':>7} | {'Step gain':>9}")
    print(f"{'-'*60}")
    for size in sizes:
        b = base_results[size]
        f = ft_results[size]
        opt = optimal[size]
        gain = b["avg_steps"] - f["avg_steps"]
        ft_eff = f["avg_steps"] / opt * 100 if f["avg_steps"] > 0 else 0
        print(
            f"{size}×{size:<3} {opt:>7.1f} | "
            f"{b['solve_pct']:>5.0f}% {b['avg_steps']:>7.1f} | "
            f"{f['solve_pct']:>5.0f}% {f['avg_steps']:>7.1f} ({ft_eff:.0f}%opt) | "
            f"{gain:>+9.1f}"
        )
    print(f"{'='*60}")


@app.local_entrypoint()
def main(episodes: int = EPISODES, sizes: str = ",".join(map(str, SIZES))):
    size_list = [int(s) for s in sizes.split(",")]
    run_eval.remote(episodes=episodes, sizes=size_list)
