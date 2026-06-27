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
    # Try "Action: <direction>" format first (ReAct output)
    match = re.search(r"action\s*:\s*([a-z]+)", text.lower())
    if match:
        word = match.group(1).strip()
        return word if word in VALID_ACTIONS else None
    # Fallback: first word
    word = re.sub(r"[^a-z]", "", text.strip().lower().split()[0])
    return word if word in VALID_ACTIONS else None


def _generate(prompt: str, model, tokenizer, max_new_tokens: int = 8) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def model_agent(state: dict, model=None, tokenizer=None) -> dict:
    grid = state["grid"]
    valid_moves = state["valid_moves"]
    history = state.get("history", [])
    pos = state["position"]

    if history:
        recent = ", ".join(f"({r},{c})" for r, c in history[-6:])
        history_line = f"Recent path: {recent}\n"
    else:
        history_line = ""

    # ReAct-style prompt: reason then act
    prompt = (
        f"You are navigating a text maze. @ is you, E is the exit, # are walls.\n\n"
        f"{grid}\n\n"
        f"Your position: row {pos[0]}, col {pos[1]}\n"
        f"{history_line}"
        f"Valid moves: {', '.join(valid_moves)}\n\n"
        f"Think:\n"
        f"1. E is at the bottom-right. Am I closer by going down or right?\n"
        f"2. Have I been looping? Avoid positions in my recent path.\n"
        f"3. Pick the valid move that makes most progress toward E.\n\n"
        f"Action: "
    )

    raw = None
    action = None

    try:
        raw = _generate(prompt, model, tokenizer, max_new_tokens=60)
        action = _parse_action(raw)
    except Exception:
        pass

    if action is None:
        fallback = (
            f"Choose one: {', '.join(valid_moves)}.\n"
            f"Reply with that single word only."
        )
        try:
            raw2 = _generate(fallback, model, tokenizer, max_new_tokens=8)
            action = _parse_action(raw2)
            if action is not None:
                raw = raw2
        except Exception:
            pass

    if action is None:
        action = random.choice(valid_moves) if valid_moves else random.choice(list(VALID_ACTIONS))

    return {"action": action, "raw_output": raw}
