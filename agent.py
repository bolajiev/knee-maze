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
    word = re.sub(r"[^a-z]", "", text.strip().lower().split()[0])
    return word if word in VALID_ACTIONS else None


def _generate(prompt: str, model, tokenizer) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=8, do_sample=False)
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def model_agent(state: dict, model=None, tokenizer=None) -> dict:
    grid = state["grid"]
    valid_moves = state["valid_moves"]

    primary = (
        f"You are navigating a text maze. Your position is @, the goal is E.\n\n"
        f"{grid}\n\n"
        f"Valid moves: {', '.join(valid_moves)}\n\n"
        f"Reply with exactly one word — up, down, left, or right."
    )

    raw = None
    action = None

    try:
        raw = _generate(primary, model, tokenizer)
        action = _parse_action(raw)
    except Exception:
        pass

    if action is None:
        strict = (
            f"Choose one: {', '.join(valid_moves)}.\n"
            f"Reply with that single word only. No punctuation, no explanation."
        )
        try:
            raw2 = _generate(strict, model, tokenizer)
            action = _parse_action(raw2)
            if action is not None:
                raw = raw2
        except Exception:
            pass

    if action is None:
        action = random.choice(valid_moves) if valid_moves else random.choice(list(VALID_ACTIONS))

    return {"action": action, "raw_output": raw}
