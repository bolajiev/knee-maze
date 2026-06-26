"""
Generate SFT training dataset from BFS-optimal maze trajectories.

Each training example is one step along the optimal path:
  user:      the exact prompt model_agent sends at inference time
  assistant: the BFS-optimal direction

Usage:
    python generate_dataset.py --n-mazes 10000 --maze-size 8 --output sft_train.jsonl
    python generate_dataset.py --n-mazes 10000 --maze-size 8 --push
"""
import argparse
import json
import os
import random

from maze import DIRECTIONS, generate_maze, render
from solver import solve_maze


def make_user_prompt(grid: str, valid_moves: list[str]) -> str:
    return (
        f"You are navigating a text maze. Your position is @, the goal is E.\n\n"
        f"{grid}\n\n"
        f"Valid moves: {', '.join(valid_moves)}\n\n"
        f"Reply with exactly one word — up, down, left, or right."
    )


def generate_examples(n_mazes: int, maze_size: int, seed_offset: int = 0) -> list[dict]:
    examples = []
    skipped = 0

    for i in range(n_mazes):
        seed = seed_offset + i
        maze = generate_maze(maze_size, seed)
        path = solve_maze(maze)

        if not path:
            skipped += 1
            continue

        pos = maze.start
        for direction in path:
            grid = render(maze, pos)
            valid_moves = maze.valid_moves(pos)

            examples.append({
                "messages": [
                    {
                        "role": "user",
                        "content": make_user_prompt(grid, valid_moves),
                    },
                    {
                        "role": "assistant",
                        "content": direction,
                    },
                ]
            })

            dr, dc = DIRECTIONS[direction]
            pos = (pos[0] + dr, pos[1] + dc)

        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{n_mazes} mazes processed — {len(examples)} examples so far")

    if skipped:
        print(f"  Warning: {skipped} mazes skipped (unsolvable)")

    return examples


def save_jsonl(examples: list[dict], path: str):
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"Saved {len(examples)} examples → {path}")


def push_to_hub(local_path: str, repo_id: str, path_in_repo: str):
    from huggingface_hub import HfApi
    token = os.getenv("HF_TOKEN", "")
    if not token:
        raise SystemExit("HF_TOKEN not set")
    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="dataset",
    )
    print(f"Uploaded → {repo_id}/{path_in_repo}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-mazes", type=int, default=10_000)
    p.add_argument("--maze-size", type=int, default=8)
    p.add_argument("--seed-offset", type=int, default=0,
                   help="Start seed (use a non-zero offset to avoid overlap with Phase 1 test mazes)")
    p.add_argument("--output", default="sft_train.jsonl")
    p.add_argument("--push", action="store_true",
                   help="Push to HF Dataset repo after generating")
    p.add_argument("--repo-id", default="bolajiev/knee-maze-logs")
    p.add_argument("--path-in-repo", default="sft/train.jsonl")
    args = p.parse_args()

    print(f"Generating {args.n_mazes} mazes (size {args.maze_size}x{args.maze_size}), seeds {args.seed_offset}–{args.seed_offset + args.n_mazes - 1}")
    examples = generate_examples(args.n_mazes, args.maze_size, args.seed_offset)

    # Shuffle so the trainer sees varied mazes across batches
    random.shuffle(examples)

    save_jsonl(examples, args.output)

    avg_steps = len(examples) / args.n_mazes
    print(f"Stats: {len(examples)} total examples, avg {avg_steps:.1f} steps/maze")

    if args.push:
        push_to_hub(args.output, args.repo_id, args.path_in_repo)


if __name__ == "__main__":
    main()
