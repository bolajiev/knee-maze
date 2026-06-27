"""
Generate DPO preference pairs for Phase 3.

For each maze step on a BFS-optimal path:
  chosen  = BFS-optimal direction
  rejected = valid direction that maximises remaining BFS distance to E
             (the worst informed wrong choice)

Skips steps where there is only one valid move (no real choice to contrast).

Usage:
    python generate_dpo_dataset.py --mazes 3000 --output dpo_train.jsonl
    python generate_dpo_dataset.py --mazes 3000 --push-to-hub
"""
import argparse
import json
import random
from collections import deque

from maze import DIRECTIONS, generate_maze, render
from solver import solve_maze


def bfs_distances(maze):
    """BFS from E outward → distance dict {pos: steps_to_E}."""
    dist = {maze.end: 0}
    queue = deque([maze.end])
    while queue:
        pos = queue.popleft()
        for direction, (dr, dc) in DIRECTIONS.items():
            npos = (pos[0] + dr, pos[1] + dc)
            if npos not in dist and maze.can_move(pos, npos):
                dist[npos] = dist[pos] + 1
                queue.append(npos)
    return dist


def make_prompt(grid: str, valid_moves: list[str]) -> str:
    return (
        f"You are navigating a text maze. Your position is @, the goal is E.\n\n"
        f"{grid}\n\n"
        f"Valid moves: {', '.join(valid_moves)}\n\n"
        f"Reply with exactly one word — up, down, left, or right."
    )


def generate_pairs(num_mazes: int, start_seed: int = 10000):
    pairs = []
    for seed in range(start_seed, start_seed + num_mazes):
        maze = generate_maze(8, seed)
        optimal_path = solve_maze(maze)
        if not optimal_path:
            continue

        dist = bfs_distances(maze)
        pos = maze.start

        for optimal_dir in optimal_path:
            valid_moves = maze.valid_moves(pos)

            if len(valid_moves) < 2:
                dr, dc = DIRECTIONS[optimal_dir]
                pos = (pos[0] + dr, pos[1] + dc)
                continue

            other_moves = [m for m in valid_moves if m != optimal_dir]
            if not other_moves:
                dr, dc = DIRECTIONS[optimal_dir]
                pos = (pos[0] + dr, pos[1] + dc)
                continue

            # Rejected = valid move that leaves us furthest from E
            def dist_after(move):
                dr, dc = DIRECTIONS[move]
                npos = (pos[0] + dr, pos[1] + dc)
                return dist.get(npos, 9999)

            rejected_dir = max(other_moves, key=dist_after)

            grid = render(maze, pos)
            prompt = make_prompt(grid, valid_moves)

            pairs.append({
                "chosen": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": optimal_dir},
                ],
                "rejected": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": rejected_dir},
                ],
            })

            dr, dc = DIRECTIONS[optimal_dir]
            pos = (pos[0] + dr, pos[1] + dc)

        if (seed - start_seed + 1) % 500 == 0:
            print(f"  {seed - start_seed + 1}/{num_mazes} mazes — {len(pairs)} pairs so far")

    return pairs


def push_to_hub(local_path: str, repo_id: str, path_in_repo: str):
    from huggingface_hub import HfApi
    import os
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN not set — skipping hub upload")
        return
    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        commit_message=f"add DPO pairs ({path_in_repo})",
    )
    print(f"Uploaded to {repo_id}/{path_in_repo}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mazes", type=int, default=3000)
    parser.add_argument("--start-seed", type=int, default=10000)
    parser.add_argument("--output", default="dpo_train.jsonl")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--repo-id", default="bolajiev/knee-maze-logs")
    parser.add_argument("--path-in-repo", default="dpo/train.jsonl")
    args = parser.parse_args()

    print(f"Generating DPO pairs from {args.mazes} mazes (seeds {args.start_seed}–{args.start_seed + args.mazes - 1})")
    pairs = generate_pairs(args.mazes, args.start_seed)
    print(f"Total pairs: {len(pairs)}")

    with open(args.output, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    print(f"Saved → {args.output}")

    if args.push_to_hub:
        push_to_hub(args.output, args.repo_id, args.path_in_repo)


if __name__ == "__main__":
    main()
