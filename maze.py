import random
from dataclasses import dataclass, field

DIRECTIONS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


@dataclass
class Maze:
    size: int
    passages: dict  # {(r, c): set of directly reachable neighbors}
    start: tuple
    end: tuple

    def can_move(self, from_pos: tuple, to_pos: tuple) -> bool:
        return to_pos in self.passages.get(from_pos, set())

    def valid_moves(self, pos: tuple) -> list:
        r, c = pos
        moves = []
        for direction, (dr, dc) in DIRECTIONS.items():
            neighbor = (r + dr, c + dc)
            if self.can_move(pos, neighbor):
                moves.append(direction)
        return moves


def generate_maze_wilson(size: int, seed: int) -> Maze:
    """Wilson's algorithm — loop-erased random walk, uniform spanning tree.
    Produces mazes with longer winding paths and fewer dead-end corridors than DFS.
    Harder to navigate: no obvious directional bias toward the exit.
    """
    rng = random.Random(seed)
    passages = {(r, c): set() for r in range(size) for c in range(size)}
    in_tree = set()
    all_cells = [(r, c) for r in range(size) for c in range(size)]

    # Seed with one cell
    in_tree.add((0, 0))

    def cell_neighbors(r, c):
        result = []
        for dr, dc in DIRECTIONS.values():
            nr, nc = r + dr, c + dc
            if 0 <= nr < size and 0 <= nc < size:
                result.append((nr, nc))
        return result

    remaining = [c for c in all_cells if c not in in_tree]
    rng.shuffle(remaining)

    for start in remaining:
        if start in in_tree:
            continue
        # Loop-erased random walk from start until we hit the tree
        path = [start]
        visited_in_walk = {start: 0}
        current = start
        while current not in in_tree:
            neighbors = cell_neighbors(*current)
            next_cell = rng.choice(neighbors)
            if next_cell in visited_in_walk:
                # Erase the loop
                loop_start = visited_in_walk[next_cell]
                path = path[:loop_start + 1]
                visited_in_walk = {c: i for i, c in enumerate(path)}
            else:
                visited_in_walk[next_cell] = len(path)
                path.append(next_cell)
            current = next_cell
        # Carve the path into the tree
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            passages[a].add(b)
            passages[b].add(a)
            in_tree.add(a)
        in_tree.add(path[-1])

    return Maze(size=size, passages=passages, start=(0, 0), end=(size - 1, size - 1))


def generate_maze(size: int, seed: int) -> Maze:
    rng = random.Random(seed)
    passages = {(r, c): set() for r in range(size) for c in range(size)}
    visited = set()

    def neighbors(r, c):
        result = []
        for dr, dc in DIRECTIONS.values():
            nr, nc = r + dr, c + dc
            if 0 <= nr < size and 0 <= nc < size:
                result.append((nr, nc))
        return result

    # Randomized iterative backtracker — guaranteed perfect maze
    start_cell = (0, 0)
    stack = [start_cell]
    visited.add(start_cell)

    while stack:
        r, c = stack[-1]
        unvisited = [(nr, nc) for nr, nc in neighbors(r, c) if (nr, nc) not in visited]
        if unvisited:
            nr, nc = rng.choice(unvisited)
            passages[(r, c)].add((nr, nc))
            passages[(nr, nc)].add((r, c))
            visited.add((nr, nc))
            stack.append((nr, nc))
        else:
            stack.pop()

    return Maze(
        size=size,
        passages=passages,
        start=(0, 0),
        end=(size - 1, size - 1),
    )


def render(maze: Maze, agent_pos: tuple) -> str:
    size = maze.size
    # Rendered grid is (2*size+1) x (2*size+1)
    # Cell (r,c) maps to rendered position (2r+1, 2c+1)
    grid = [["#"] * (2 * size + 1) for _ in range(2 * size + 1)]

    # Open cell interiors and label start/end
    for r in range(size):
        for c in range(size):
            if (r, c) == maze.start:
                char = "S"
            elif (r, c) == maze.end:
                char = "E"
            else:
                char = "."
            grid[2 * r + 1][2 * c + 1] = char

    # Open passages: wall between (r,c) and (nr,nc) is at rendered midpoint
    for (r, c), neighbors in maze.passages.items():
        for nr, nc in neighbors:
            grid[r + nr + 1][c + nc + 1] = "."

    # Agent drawn on top of whatever's underneath
    ar, ac = agent_pos
    grid[2 * ar + 1][2 * ac + 1] = "@"

    return "\n".join("".join(row) for row in grid)
