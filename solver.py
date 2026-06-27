from collections import deque

from maze import DIRECTIONS, Maze, generate_maze, render


def bfs_distance_map(maze: Maze) -> dict:
    """Reverse BFS from end. Returns {pos: steps_to_end} for every reachable cell."""
    distances = {maze.end: 0}
    queue = deque([maze.end])
    while queue:
        pos = queue.popleft()
        for dr, dc in DIRECTIONS.values():
            neighbor = (pos[0] + dr, pos[1] + dc)
            if neighbor not in distances and maze.can_move(pos, neighbor):
                distances[neighbor] = distances[pos] + 1
                queue.append(neighbor)
    return distances


def solve_maze(maze: Maze) -> list[str]:
    """
    BFS shortest path from maze.start to maze.end.
    Returns list of direction strings e.g. ["right", "down", "right", ...].
    Returns [] only if maze is unsolvable (never happens for a perfect maze).
    """
    start = maze.start
    end = maze.end

    queue = deque([(start, [])])
    visited = {start}

    while queue:
        pos, path = queue.popleft()
        if pos == end:
            return path
        for direction, (dr, dc) in DIRECTIONS.items():
            next_pos = (pos[0] + dr, pos[1] + dc)
            if next_pos not in visited and maze.can_move(pos, next_pos):
                visited.add(next_pos)
                queue.append((next_pos, path + [direction]))

    return []


if __name__ == "__main__":
    # Quick sanity check
    errors = 0
    for seed in range(200):
        m = generate_maze(8, seed)
        path = solve_maze(m)
        if not path:
            print(f"  ERROR: seed {seed} unsolvable")
            errors += 1
            continue
        # Walk the path and verify we reach the end
        pos = m.start
        for d in path:
            dr, dc = DIRECTIONS[d]
            next_pos = (pos[0] + dr, pos[1] + dc)
            assert m.can_move(pos, next_pos), f"seed {seed}: invalid move {d} at {pos}"
            pos = next_pos
        assert pos == m.end, f"seed {seed}: path ended at {pos}, not {m.end}"

    if errors == 0:
        print(f"solver OK — 200 mazes, all solved, zero errors")
        m = generate_maze(8, 42)
        path = solve_maze(m)
        print(f"Example (seed=42): {len(path)} steps — {path}")
        print(render(m, m.start))
