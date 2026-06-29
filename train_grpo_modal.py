"""
Phase 4b: GRPO fine-tuning with BFS oracle as dense per-step reward.

Research goal: push BFS-optimal decision rate from ~78% (SFT) toward 95%+.
Each training example is a single maze step. Reward directly scores whether
the model's chosen move reduces BFS distance.

Reward per step:
  +1.0   move reduces BFS distance (BFS-optimal)
   0.0   move maintains BFS distance (lateral corridor)
  -0.5   move increases BFS distance (wrong direction)
  +5.0   bonus: reached exit
  -1.0   wall hit (invalid move)
  -0.3   unparseable output

Starts from bolajiev/qwen-maze-traces (SFT checkpoint).
Pushes to bolajiev/qwen-maze-grpo.

Usage:
    .venv/bin/modal run train_grpo_modal.py
"""
import os
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.6.0",
        "transformers>=4.45.0,<5.0.0",
        "trl>=1.1.0,<1.7.0",
        "peft>=0.13.0",
        "accelerate>=0.34.0",
        "datasets>=2.21.0",
        "huggingface_hub>=0.25.0",
    )
)

app = modal.App("knee-maze-grpo", image=image)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

SFT_MODEL      = "bolajiev/qwen-maze-traces"
OUTPUT_REPO    = "bolajiev/qwen-maze-grpo"

LEARNING_RATE  = 5e-6
NUM_EPOCHS     = 1
BATCH_SIZE     = 4
GRAD_ACCUM     = 4
MAX_NEW_TOKENS = 8       # action-first format — word is in first 2-3 tokens
N_GENERATIONS  = 4       # GRPO rollouts per prompt
MAX_EXAMPLES   = 10000
MAZE_SIZES     = [6, 7, 8, 11]   # matches SFT curriculum
SEED_OFFSET    = 50000           # separate from SFT seeds (20000-23000)


@app.function(
    gpu="A10G",
    timeout=14400,
    secrets=[modal.Secret.from_name("hf-token")],
    volumes={"/root/.cache/huggingface": hf_cache},
)
def train():
    import re
    import random
    from collections import deque
    from dataclasses import dataclass

    import torch
    from datasets import Dataset
    from huggingface_hub import HfApi
    from peft import LoraConfig, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    hf_token = os.environ["HF_TOKEN"]
    api = HfApi(token=hf_token)

    # ── Inline maze code ─────────────────────────────────────────────────────
    DIRECTIONS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
    VALID_ACTIONS = {"up", "down", "left", "right"}

    @dataclass
    class Maze:
        size: int
        passages: dict
        start: tuple
        end: tuple
        def can_move(self, f, t): return t in self.passages.get(f, set())
        def valid_moves(self, pos):
            r, c = pos
            return [d for d, (dr, dc) in DIRECTIONS.items()
                    if self.can_move(pos, (r + dr, c + dc))]

    def generate_maze(size, seed):
        rng = random.Random(seed)
        passages = {(r, c): set() for r in range(size) for c in range(size)}
        visited = set()
        def nbrs(r, c):
            return [(r+dr, c+dc) for dr, dc in DIRECTIONS.values()
                    if 0 <= r+dr < size and 0 <= c+dc < size]
        stack = [(0, 0)]; visited.add((0, 0))
        while stack:
            r, c = stack[-1]
            unvis = [(nr, nc) for nr, nc in nbrs(r, c) if (nr, nc) not in visited]
            if unvis:
                nr, nc = rng.choice(unvis)
                passages[(r,c)].add((nr,nc)); passages[(nr,nc)].add((r,c))
                visited.add((nr,nc)); stack.append((nr,nc))
            else:
                stack.pop()
        return Maze(size, passages, (0, 0), (size-1, size-1))

    def bfs_distance_map(maze):
        dist = {maze.end: 0}
        q = deque([maze.end])
        while q:
            pos = q.popleft()
            for dr, dc in DIRECTIONS.values():
                nb = (pos[0]+dr, pos[1]+dc)
                if nb not in dist and maze.can_move(pos, nb):
                    dist[nb] = dist[pos] + 1; q.append(nb)
        return dist

    def solve_maze(maze):
        bfs_map = bfs_distance_map(maze)
        pos, path = maze.start, []
        visited = {pos}
        for _ in range(maze.size * maze.size * 4):
            if pos == maze.end: return path
            moves = maze.valid_moves(pos)
            best = min(moves, key=lambda m: bfs_map.get(
                (pos[0]+DIRECTIONS[m][0], pos[1]+DIRECTIONS[m][1]), 9999))
            dr, dc = DIRECTIONS[best]
            pos = (pos[0]+dr, pos[1]+dc)
            if pos in visited: return None
            visited.add(pos); path.append(best)
        return None

    def parse_action(text):
        if not text: return None
        m = re.search(r"action\s*:\s*([a-z]+)", text.lower())
        if m:
            w = m.group(1).strip()
            return w if w in VALID_ACTIONS else None
        parts = text.strip().lower().split()
        if parts:
            w = re.sub(r"[^a-z]", "", parts[0])
            return w if w in VALID_ACTIONS else None
        return None

    def make_prompt(maze, pos, bfs_map, history=None):
        r, c = pos
        gr, gc = maze.end
        rows_to_exit = gr - r
        cols_to_exit = gc - c
        bfs_dist = bfs_map.get(pos, -1)
        valid_moves = maze.valid_moves(pos)
        walls = {d: not maze.can_move(pos, (r+dr, c+dc))
                 for d, (dr, dc) in DIRECTIONS.items()}
        wall_parts = [f"{d}={'blocked' if walls[d] else 'open'}"
                      for d in ("up", "down", "left", "right")]
        history_line = ""
        if history:
            recent = " → ".join(f"({pr},{pc})" for pr, pc in history[-6:])
            history_line = f"Recent path: {recent}\n"
            if pos in (history or [])[-4:]:
                history_line += "WARNING: looping detected — do not go back the way you came.\n"
        return (
            f"Maze ({maze.size}×{maze.size}). @ = you, E = exit.\n\n"
            f"Rows to exit: {rows_to_exit}  |  Cols to exit: {cols_to_exit}  |  BFS steps to exit: {bfs_dist}\n"
            f"Walls: {', '.join(wall_parts)}\n"
            f"Valid moves: {', '.join(valid_moves)}\n"
            f"{history_line}\n"
            f"Think:\n"
            f"1. Which valid move reduces BFS distance?\n"
            f"2. Avoid moves that return to recently visited positions.\n"
            f"3. State the best move.\n\n"
            f"Action: "
        )

    # ── Reward function ───────────────────────────────────────────────────────
    def maze_reward(prompts, completions, maze_seed, position_r, position_c,
                    maze_size, bfs_dist_before, **kwargs):
        rewards = []
        for i, completion in enumerate(completions):
            seed  = int(maze_seed[i])
            size  = int(maze_size[i])
            pos   = (int(position_r[i]), int(position_c[i]))
            d_before = int(bfs_dist_before[i])

            maze    = generate_maze(size, seed)
            bfs_map = bfs_distance_map(maze)

            # Extract action from completion
            text   = completion[0]["content"] if isinstance(completion, list) else completion
            action = parse_action(text)

            if action is None:
                rewards.append(-0.3); continue

            dr, dc   = DIRECTIONS[action]
            next_pos = (pos[0]+dr, pos[1]+dc)

            if not maze.can_move(pos, next_pos):
                rewards.append(-1.0); continue

            if next_pos == maze.end:
                rewards.append(5.0); continue

            d_after     = bfs_map.get(next_pos, d_before + 1)
            improvement = d_before - d_after   # +1 closer, 0 same, -1 farther
            if improvement > 0:
                rewards.append(1.0)
            elif improvement == 0:
                rewards.append(0.0)
            else:
                rewards.append(-0.5)

        return rewards

    # ── Build dataset — sample positions across full paths ───────────────────
    # Sample from random positions in the maze path, not just start position.
    # This gives the model training signal at harder mid-maze decision points.
    print("Building GRPO dataset (positions across full paths)...")
    rng = random.Random(42)
    grpo_rows = []
    i = 0
    while len(grpo_rows) < MAX_EXAMPLES:
        seed = SEED_OFFSET + i
        i += 1
        size = rng.choice(MAZE_SIZES)
        maze = generate_maze(size, seed)
        bfs_map = bfs_distance_map(maze)
        path = solve_maze(maze)
        if not path or len(path) < 5:
            continue

        # Walk the path and sample up to 8 positions per maze
        pos = maze.start
        history = []
        sample_positions = sorted(rng.sample(range(len(path)), min(8, len(path))))
        step_idx = 0
        for step_num, action in enumerate(path):
            if step_idx < len(sample_positions) and step_num == sample_positions[step_idx]:
                prompt = make_prompt(maze, pos, bfs_map, list(history[-6:]))
                grpo_rows.append({
                    "prompt": [{"role": "user", "content": prompt}],
                    "maze_seed": seed,
                    "maze_size": size,
                    "position_r": pos[0],
                    "position_c": pos[1],
                    "bfs_dist_before": bfs_map.get(pos, 0),
                })
                step_idx += 1
            history.append(pos)
            dr, dc = DIRECTIONS[action]
            pos = (pos[0]+dr, pos[1]+dc)

        if len(grpo_rows) % 1000 == 0:
            print(f"  {len(grpo_rows)} examples...")

    grpo_rows = grpo_rows[:MAX_EXAMPLES]
    rng.shuffle(grpo_rows)
    dataset = Dataset.from_list(grpo_rows)
    print(f"GRPO dataset: {len(dataset)} examples across path positions")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Loading {SFT_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = 512
    tokenizer.truncation_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        SFT_MODEL, torch_dtype=torch.bfloat16, device_map={"": 0}, token=hf_token,
    )

    lora_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM,
    )

    training_args = GRPOConfig(
        output_dir="/tmp/grpo-output",
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        bf16=True,
        logging_steps=25,
        save_strategy="epoch",
        report_to="none",
        num_generations=N_GENERATIONS,
        max_completion_length=MAX_NEW_TOKENS,
        temperature=0.8,    # must be > 0 so 4 rollouts produce different outputs
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        reward_funcs=[maze_reward],
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    print("Training GRPO...")
    trainer.train()

    print("Merging LoRA weights...")
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained("/tmp/grpo-merged")
    tokenizer.save_pretrained("/tmp/grpo-merged")

    print(f"Pushing to {OUTPUT_REPO}...")
    api.create_repo(OUTPUT_REPO, repo_type="model", private=True, exist_ok=True, token=hf_token)
    api.upload_folder(
        folder_path="/tmp/grpo-merged",
        repo_id=OUTPUT_REPO, repo_type="model", token=hf_token,
    )
    print(f"Done — huggingface.co/{OUTPUT_REPO}")


@app.local_entrypoint()
def main():
    train.remote()
