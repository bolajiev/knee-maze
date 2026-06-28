"""
Phase 4a: Generate SFT training data with full reasoning traces.

Instead of (state → direction), each example is:
  user:      structured state (position, walls, BFS dist, history)
  assistant: step-by-step reasoning + Action: <direction>

Key quality levers (from Searchformer ablations):
  1. Traces teach the algorithm, not just correct answers
  2. Hard mazes only (path_length >= MIN_PATH_LENGTH) — easy mazes dilute signal
  3. Mix DFS + Wilson's generators — DFS has directional bias, Wilson's (uniform
     spanning tree) produces longer winding paths with no bias, forces real search

Usage:
    python generate_sft_traces.py --n-mazes 3000 --push
"""
import argparse
import json
import os
import random

from maze import DIRECTIONS, generate_maze, generate_maze_wilson
from solver import bfs_distance_map, solve_maze
from agent import make_structured_prompt

MIN_PATH_LENGTH = 8   # 6×6 mazes average ~15 steps but some are shorter — keep anything requiring real search


def _walls_at(maze, pos):
    r, c = pos
    return {
        d: not maze.can_move(pos, (r + dr, c + dc))
        for d, (dr, dc) in DIRECTIONS.items()
    }


def _build_reasoning(maze, pos, valid_moves, bfs_map, optimal_action):
    """Programmatically generate a BFS reasoning trace for one step."""
    r, c = pos
    gr, gc = maze.end
    dist_here = bfs_map.get(pos, 0)
    rows_to_exit = gr - r
    cols_to_exit = gc - c
    lines = []

    lines.append(f"{rows_to_exit} rows and {cols_to_exit} cols to exit. BFS distance: {dist_here} steps.")

    # Evaluate each valid move
    move_evals = []
    for move in valid_moves:
        dr, dc = DIRECTIONS[move]
        next_pos = (r + dr, c + dc)
        next_dist = bfs_map.get(next_pos, dist_here + 999)
        delta = dist_here - next_dist  # positive = closer, negative = farther
        move_evals.append((move, next_pos, next_dist, delta))

    move_evals.sort(key=lambda x: x[3], reverse=True)

    for move, next_pos, next_dist, delta in move_evals:
        tag = "closer" if delta > 0 else ("same" if delta == 0 else "farther")
        lines.append(f"  {move} → ({next_pos[0]},{next_pos[1]}): BFS dist {next_dist} [{delta:+d}, {tag}]")

    best_move, best_pos, best_dist, best_delta = move_evals[0]
    if best_delta > 0:
        lines.append(f"{best_move} reduces BFS distance by {best_delta}. Best move: {best_move}.")
    elif best_delta == 0:
        lines.append(f"All moves maintain BFS distance. Choosing {best_move}.")
    else:
        lines.append(f"All moves increase distance (corridor backtrack). Choosing {best_move} (least bad).")

    return "\n".join(lines)


def make_trace_example(maze, pos, valid_moves, bfs_map, optimal_action, history):
    """Build one training example with reasoning trace."""
    state = {
        "position": pos,
        "valid_moves": valid_moves,
        "walls": _walls_at(maze, pos),
        "bfs_dist": bfs_map.get(pos, -1),
        "goal": maze.end,
        "maze_size": maze.size,
        "history": history,
    }
    user_prompt = make_structured_prompt(state)
    # Strip the trailing "Action: " from the prompt since it goes in assistant turn
    user_prompt = user_prompt.rstrip()
    if user_prompt.endswith("Action:"):
        user_prompt = user_prompt[:-len("Action:")].rstrip()

    reasoning = _build_reasoning(maze, pos, valid_moves, bfs_map, optimal_action)
    assistant_response = f"{reasoning}\n\nAction: {optimal_action}"

    return {
        "messages": [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_response},
        ]
    }


def generate_examples(n_mazes: int, seed_offset: int = 20000) -> list[dict]:
    # Curriculum: 6×6 through 11×11 — covers the full range users test on the Space
    # Generator mix: 60% DFS (directional bias), 40% Wilson's (uniform, no bias, harder)
    size_dist  = [(6, 0.15), (7, 0.15), (8, 0.40), (11, 0.30)]
    gen_dist   = [("dfs", 0.60), ("wilson", 0.40)]
    rng = random.Random(42)
    examples = []
    skipped_short = 0
    skipped_unsolvable = 0
    attempted = 0

    accepted = 0
    i = 0
    while accepted < n_mazes:
        seed = seed_offset + i
        i += 1
        attempted += 1

        # Sample size
        roll = rng.random()
        cumulative = 0.0
        maze_size = 8
        for size, prob in size_dist:
            cumulative += prob
            if roll < cumulative:
                maze_size = size
                break

        # Sample generator
        roll2 = rng.random()
        cumulative2 = 0.0
        gen = "dfs"
        for name, prob in gen_dist:
            cumulative2 += prob
            if roll2 < cumulative2:
                gen = name
                break

        maze = generate_maze(maze_size, seed) if gen == "dfs" else generate_maze_wilson(maze_size, seed)
        path = solve_maze(maze)

        if not path:
            skipped_unsolvable += 1
            continue

        bfs_map = bfs_distance_map(maze)
        path_len = bfs_map.get(maze.start, 0)

        # Skip easy mazes — no real search needed, just noise
        if path_len < MIN_PATH_LENGTH:
            skipped_short += 1
            continue

        pos = maze.start
        history = []

        for optimal_action in path:
            valid_moves = maze.valid_moves(pos)
            ex = make_trace_example(maze, pos, valid_moves, bfs_map, optimal_action, list(history[-6:]))
            examples.append(ex)
            history.append(pos)
            dr, dc = DIRECTIONS[optimal_action]
            pos = (pos[0] + dr, pos[1] + dc)

        accepted += 1
        if accepted % 500 == 0:
            print(f"  {accepted}/{n_mazes} mazes — {len(examples)} examples (skipped {skipped_short} short)")

    print(f"Done: {attempted} attempted, {skipped_short} skipped (too short), {len(examples)} examples from hard mazes")
    return examples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-mazes", type=int, default=3_000,
                   help="Target number of accepted hard mazes (path >= 15 steps)")
    p.add_argument("--seed-offset", type=int, default=20000,
                   help="Use seeds 20k+ to avoid overlap with SFT/DPO data")
    p.add_argument("--output", default="sft_traces.jsonl")
    p.add_argument("--push", action="store_true")
    p.add_argument("--repo-id", default="bolajiev/knee-maze-logs")
    p.add_argument("--path-in-repo", default="sft_traces/train.jsonl")
    args = p.parse_args()

    print(f"Targeting {args.n_mazes} hard mazes (path_length >= {MIN_PATH_LENGTH})")
    print("Sizes: 60% 8×8, 40% 11×11  |  Generators: 60% DFS, 40% Wilson's")
    examples = generate_examples(args.n_mazes, args.seed_offset)

    random.shuffle(examples)
    with open(args.output, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"Saved {len(examples)} examples → {args.output}")

    if args.push:
        from huggingface_hub import HfApi
        token = os.getenv("HF_TOKEN", "")
        if not token:
            raise SystemExit("HF_TOKEN not set")
        api = HfApi(token=token)
        api.upload_file(
            path_or_fileobj=args.output,
            path_in_repo=args.path_in_repo,
            repo_id=args.repo_id,
            repo_type="dataset",
        )
        print(f"Uploaded → {args.repo_id}/{args.path_in_repo}")


if __name__ == "__main__":
    main()
