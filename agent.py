import random
import re

VALID_ACTIONS = {"up", "down", "left", "right"}


def random_agent(state: dict, model=None, tokenizer=None) -> dict:
    valid_moves = state["valid_moves"]
    action = random.choice(valid_moves) if valid_moves else random.choice(list(VALID_ACTIONS))
    return {"action": action, "raw_output": None}


def _parse_action(text: str) -> str | None:
    if not text:
        return None
    # "Action: down" format first
    match = re.search(r"action\s*:\s*([a-z]+)", text.lower())
    if match:
        word = match.group(1).strip()
        return word if word in VALID_ACTIONS else None
    # fallback: first word
    word = re.sub(r"[^a-z]", "", text.strip().lower().split()[0])
    return word if word in VALID_ACTIONS else None


def _generate(prompt: str, model, tokenizer, max_new_tokens: int = 8) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def make_structured_prompt(state: dict) -> str:
    """Build structured state prompt — no raw ASCII grid, explicit BFS distance."""
    pos = state["position"]
    valid_moves = state["valid_moves"]
    walls = state.get("walls", {})
    bfs_dist = state.get("bfs_dist", "?")
    goal = state.get("goal", "?")
    maze_size = state.get("maze_size", "?")
    history = state.get("history", [])

    r, c = pos
    gr, gc = goal if goal != "?" else (r + 1, c + 1)
    rows_to_exit = gr - r
    cols_to_exit = gc - c

    wall_parts = [
        f"{d}={'blocked' if walls.get(d, True) else 'open'}"
        for d in ("up", "down", "left", "right")
    ]

    if history:
        looping = len(history) >= 2 and pos in history[-4:]
        recent = " → ".join(f"({pr},{pc})" for pr, pc in history[-6:])
        history_line = f"Recent path: {recent}\n"
        if looping:
            history_line += "WARNING: looping detected — do not go back the way you came.\n"
    else:
        history_line = ""

    return (
        f"Maze ({maze_size}×{maze_size}). @ = you, E = exit.\n\n"
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


def model_agent(state: dict, model=None, tokenizer=None) -> dict:
    prompt = make_structured_prompt(state)
    raw = None
    action = None

    try:
        raw = _generate(prompt, model, tokenizer, max_new_tokens=20)
        action = _parse_action(raw)
    except Exception:
        pass

    if action is None:
        valid_moves = state["valid_moves"]
        fallback = f"Choose one: {', '.join(valid_moves)}. Reply with that word only."
        try:
            raw2 = _generate(fallback, model, tokenizer, max_new_tokens=8)
            action = _parse_action(raw2)
            if action is not None:
                raw = raw2
        except Exception:
            pass

    if action is None:
        valid_moves = state["valid_moves"]
        action = random.choice(valid_moves) if valid_moves else random.choice(list(VALID_ACTIONS))

    return {"action": action, "raw_output": raw}
