import random
from datetime import datetime, timezone

from maze import DIRECTIONS, generate_maze, render
from solver import bfs_distance_map


def run_episode(
    agent_fn,
    maze_size: int,
    max_steps: int,
    seed: int,
    episode_id: int = 0,
    run_id: str = "",
    agent_type: str = "unknown",
    model=None,
    tokenizer=None,
):
    """
    Generator — yields one snapshot dict per step, then a final summary snapshot.

    Each yield:
        {
            "grid":             str,   # rendered maze after this step
            "step_record":      dict,  # full log record for this step
            "done":             bool,  # True on the last step of the episode
            "episode_summary":  dict | None,  # only present when done=True
        }

    Caller collects step_records + episode_summary for logging, and uses
    "grid" to update the UI or print to console.
    """
    maze = generate_maze(maze_size, seed)
    bfs_map = bfs_distance_map(maze)
    pos = maze.start
    wall_hits = 0
    reason_ended = "timeout"
    step_records = []
    history = []  # recent positions for loop-avoidance hint

    for step in range(1, max_steps + 1):
        valid_moves = maze.valid_moves(pos)
        grid_before = render(maze, pos)
        r, c = pos
        walls = {
            d: not maze.can_move(pos, (r + dr, c + dc))
            for d, (dr, dc) in DIRECTIONS.items()
        }
        state = {
            "grid": grid_before,
            "position": pos,
            "valid_moves": valid_moves,
            "history": list(history[-10:]),
            "walls": walls,
            "bfs_dist": bfs_map.get(pos, -1),
            "goal": maze.end,
            "maze_size": maze_size,
        }

        result = agent_fn(state, model=model, tokenizer=tokenizer)
        intended_action = result["action"]
        raw_output = result.get("raw_output")

        # If the model picked a wall direction, fall back to random valid move.
        # We still log the intended action so the training signal isn't lost.
        dr, dc = DIRECTIONS[intended_action]
        if maze.can_move(pos, (pos[0] + dr, pos[1] + dc)):
            action = intended_action
            wall_hit_recorded = False
        else:
            wall_hits += 1
            wall_hit_recorded = True
            action = random.choice(valid_moves) if valid_moves else intended_action

        # Anti-oscillation: if action revisits a recent position, override with
        # the best non-revisiting valid move (sorted by BFS distance).
        # This prevents right→left→right loops that waste all 200 steps.
        recent = set(history[-6:])
        r2, c2 = pos
        dr2, dc2 = DIRECTIONS[action]
        would_revisit = (r2 + dr2, c2 + dc2) in recent
        if would_revisit and not wall_hit_recorded:
            alternatives = []
            for m in valid_moves:
                dr3, dc3 = DIRECTIONS[m]
                np = (r2 + dr3, c2 + dc3)
                if np not in recent:
                    alternatives.append((m, bfs_map.get(np, 9999)))
            if alternatives:
                action = min(alternatives, key=lambda x: x[1])[0]

        pos_before = pos
        dr, dc = DIRECTIONS[action]
        next_pos = (pos[0] + dr, pos[1] + dc)
        valid_move = maze.can_move(pos, next_pos)

        if valid_move:
            pos = next_pos

        history.append(pos_before)

        won = valid_move and pos == maze.end
        if won:
            status = "win"
            reason_ended = "win"
        elif step == max_steps:
            # Final step always closes the episode, even if it was also a wall hit
            status = "timeout"
        elif wall_hit_recorded:
            status = "wall_hit"
        else:
            status = "ongoing"

        step_record = {
            "record_type": "step",
            "run_id": run_id,
            "episode_id": episode_id,
            "step": step,
            "maze_seed": seed,
            "agent_type": agent_type,
            "position_before": list(pos_before),
            "intended_action": intended_action,   # what the model said
            "action": action,                      # what was actually executed
            "wall_hit": wall_hit_recorded,
            "valid_move": valid_move,
            "position_after": list(pos),
            "status": status,
            "raw_model_output": raw_output,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        step_records.append(step_record)

        done = status in ("win", "timeout")
        episode_summary = None

        if done:
            episode_summary = {
                "record_type": "episode_summary",
                "run_id": run_id,
                "episode_id": episode_id,
                "maze_seed": seed,
                "agent_type": agent_type,
                "maze_size": maze_size,
                "solved": reason_ended == "win",
                "total_steps": step,
                "wall_hits": wall_hits,
                "reason_ended": reason_ended,
            }

        yield {
            "grid": render(maze, pos),
            "step_record": step_record,
            "done": done,
            "episode_summary": episode_summary,
        }

        if done:
            break
