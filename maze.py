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
