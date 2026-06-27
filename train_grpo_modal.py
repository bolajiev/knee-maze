"""
Phase 4b: GRPO fine-tuning with BFS oracle as reward.

Reward function (per step):
  - Reached exit:         +10.0
  - BFS distance improved: +(improvement / dist_before)   e.g. 1 step closer on dist=10 → +0.1
  - Wall hit:             -1.0
  - Invalid parse:        -0.5

Starts from bolajiev/qwen-maze-traces (Phase 4a SFT checkpoint).
Pushes to bolajiev/qwen-maze-grpo.

Usage:
    modal run train_grpo_modal.py
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

SFT_MODEL     = "bolajiev/qwen-maze-traces"   # Phase 4a checkpoint
OUTPUT_REPO   = "bolajiev/qwen-maze-grpo"
DATASET_REPO  = "bolajiev/knee-maze-logs"
DATASET_FILE  = "sft_traces/train.jsonl"

LEARNING_RATE  = 5e-6
NUM_EPOCHS     = 1
BATCH_SIZE     = 4
GRAD_ACCUM     = 4
MAX_NEW_TOKENS = 80
N_GENERATIONS  = 4   # GRPO rollouts per prompt
MAX_EXAMPLES   = 8000


@app.function(
    gpu="T4",
    timeout=14400,
    secrets=[modal.Secret.from_name("hf-token")],
    volumes={"/root/.cache/huggingface": hf_cache},
)
def train():
    import json
    import re
    import sys
    from collections import deque

    import torch
    from datasets import Dataset
    from huggingface_hub import HfApi
    from peft import LoraConfig, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    hf_token = os.environ["HF_TOKEN"]
    api = HfApi(token=hf_token)

    # ── inline maze code so Modal doesn't need local files ──────────────────
    import random
    from dataclasses import dataclass, field as dc_field

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
            return [d for d, (dr, dc) in DIRECTIONS.items() if self.can_move(pos, (r+dr, c+dc))]

    def generate_maze(size, seed):
        rng = random.Random(seed)
        passages = {(r, c): set() for r in range(size) for c in range(size)}
        visited = set()
        def neighbors(r, c):
            return [(r+dr, c+dc) for dr, dc in DIRECTIONS.values()
                    if 0 <= r+dr < size and 0 <= c+dc < size]
        stack = [(0, 0)]
        visited.add((0, 0))
        while stack:
            r, c = stack[-1]
            unvis = [(nr, nc) for nr, nc in neighbors(r, c) if (nr, nc) not in visited]
            if unvis:
                nr, nc = rng.choice(unvis)
                passages[(r, c)].add((nr, nc)); passages[(nr, nc)].add((r, c))
                visited.add((nr, nc)); stack.append((nr, nc))
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
                    dist[nb] = dist[pos] + 1
                    q.append(nb)
        return dist

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

    # ── reward function ──────────────────────────────────────────────────────
    def maze_reward(prompts, completions, maze_seed, position_r, position_c, maze_size, bfs_dist_before, **kwargs):
        rewards = []
        for i, completion in enumerate(completions):
            seed = int(maze_seed[i])
            size = int(maze_size[i])
            pos = (int(position_r[i]), int(position_c[i]))
            dist_before = int(bfs_dist_before[i])

            maze = generate_maze(size, seed)
            bfs_map = bfs_distance_map(maze)

            action = parse_action(completion)
            if action is None:
                rewards.append(-0.5)
                continue

            dr, dc = DIRECTIONS[action]
            next_pos = (pos[0]+dr, pos[1]+dc)
            if not maze.can_move(pos, next_pos):
                rewards.append(-1.0)
                continue

            if next_pos == maze.end:
                rewards.append(10.0)
            else:
                dist_after = bfs_map.get(next_pos, dist_before + 1)
                improvement = dist_before - dist_after
                reward = improvement / max(dist_before, 1)
                rewards.append(float(reward))

        return rewards

    # ── dataset ──────────────────────────────────────────────────────────────
    print("Downloading dataset...")
    api.hf_hub_download(
        repo_id=DATASET_REPO, filename=DATASET_FILE,
        repo_type="dataset", local_dir="/tmp/dataset", token=hf_token,
    )

    rows = []
    with open(f"/tmp/dataset/{DATASET_FILE}") as f:
        for line in f:
            ex = json.loads(line)
            rows.append(ex)
    print(f"Loaded {len(rows)} examples, using first {MAX_EXAMPLES}")
    rows = rows[:MAX_EXAMPLES]

    # GRPO needs: prompt + metadata columns for reward function
    # Parse maze context from the user message (it's embedded in the structured prompt)
    grpo_rows = []
    for ex in rows:
        user_msg = ex["messages"][0]["content"]
        # Extract: Position: (r,c)  |  Exit: (gr,gc)  |  BFS steps to exit: d
        pos_m = re.search(r"Position:\s*\((\d+),(\d+)\)", user_msg)
        exit_m = re.search(r"Exit:\s*\((\d+),(\d+)\)", user_msg)
        bfs_m = re.search(r"BFS steps to exit:\s*(\d+)", user_msg)
        size_m = re.search(r"Maze \((\d+)×\d+\)", user_msg)
        # Seed is not stored in prompt — use a placeholder; GRPO will regenerate mazes at reward time
        # We encode seed in a separate column from the original dataset (not available here without regen)
        # Skip rows where we can't parse context
        if not (pos_m and exit_m and bfs_m and size_m):
            continue
        grpo_rows.append({
            "prompt": [{"role": "user", "content": user_msg}],
            "position_r": int(pos_m.group(1)),
            "position_c": int(pos_m.group(2)),
            "bfs_dist_before": int(bfs_m.group(1)),
            "maze_size": int(size_m.group(1)),
            "maze_seed": 0,  # will be set below via re-gen
        })

    # We need seeds — re-generate from the SFT traces dataset which has seeds
    # For now: use position + size to find the right seed (deterministic maze)
    # Actually: build a lookup. Generate mazes until end matches the exit coords.
    # Faster: embed seed in sft_traces output. For this first run, skip re-linking
    # and use a simpler approach: regenerate prompts directly from seeds.

    # Generate GRPO prompts directly from seeds (cleaner than parsing)
    grpo_rows = []
    print("Building GRPO dataset from maze seeds...")
    MAZE_SIZES = [5, 8, 11]
    rng_g = random.Random(99)
    for i in range(MAX_EXAMPLES):
        seed = 20000 + i
        size = rng_g.choice(MAZE_SIZES)
        maze = generate_maze(size, seed)
        bfs_map = bfs_distance_map(maze)
        path_len = bfs_map.get(maze.start, 0)
        if path_len == 0:
            continue
        pos = maze.start
        valid_moves = maze.valid_moves(pos)
        walls = {d: not maze.can_move(pos, (pos[0]+dr, pos[1]+dc))
                 for d, (dr, dc) in DIRECTIONS.items()}
        user_prompt = (
            f"Maze ({size}×{size}). @ = you, E = exit.\n\n"
            f"Position: ({pos[0]},{pos[1]})  |  Exit: ({maze.end[0]},{maze.end[1]})  |  BFS steps to exit: {path_len}\n"
            f"Walls: {', '.join(f'{d}={\"blocked\" if walls[d] else \"open\"}' for d in [\"up\",\"down\",\"left\",\"right\"])}\n"
            f"Valid moves: {', '.join(valid_moves)}\n\n"
            f"Think:\n"
            f"1. Which valid move reduces BFS distance toward ({maze.end[0]},{maze.end[1]})?\n"
            f"2. Avoid recently visited positions if looping.\n"
            f"3. State the best move.\n"
        )
        grpo_rows.append({
            "prompt": [{"role": "user", "content": user_prompt}],
            "maze_seed": seed,
            "maze_size": size,
            "position_r": pos[0],
            "position_c": pos[1],
            "bfs_dist_before": path_len,
        })

    dataset = Dataset.from_list(grpo_rows)
    print(f"GRPO dataset: {len(dataset)} examples")

    # ── model ────────────────────────────────────────────────────────────────
    print(f"Loading {SFT_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = 512
    tokenizer.truncation_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        SFT_MODEL, torch_dtype=torch.float16, device_map={"": 0}, token=hf_token,
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
        fp16=True,
        logging_steps=50,
        save_strategy="epoch",
        report_to="none",
        num_generations=N_GENERATIONS,
        max_completion_length=MAX_NEW_TOKENS,
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
    print(f"Done — {OUTPUT_REPO}")


@app.local_entrypoint()
def main():
    train.remote()
