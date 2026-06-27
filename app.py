import html
import random
import time
import uuid

import gradio as gr

from agent import model_agent
from config import BASE_MODEL, DATASET_REPO_ID, FINE_TUNED_MODEL_PATH
from logger import flush_to_dataset
from model_loader import load_model
from runner import run_episode

# Check fine-tuned model availability at startup (non-fatal)
_FINETUNED_AVAILABLE = False
if FINE_TUNED_MODEL_PATH is not None:
    try:
        load_model(FINE_TUNED_MODEL_PATH)
        _FINETUNED_AVAILABLE = True
    except Exception:
        pass


_MAZE_STYLE = (
    "font-family:'Courier New',Courier,monospace;"
    "font-size:18px;"
    "line-height:1.4;"
    "background:#0d1117;"
    "padding:14px 16px;"
    "border-radius:8px;"
    "margin:0;"
    "display:inline-block;"
    "min-width:100%;"
    "border:2px solid transparent;"
)
_MAZE_STYLE_HIT = _MAZE_STYLE.replace(
    "border:2px solid transparent;",
    "border:2px solid #f85149;box-shadow:0 0 8px #f85149;"
)

_WALL_HTML  = "<span style='color:#484f58'>█</span>"
_PATH_HTML  = "<span style='color:#0d1117'> </span>"
_COLORS = {
    "S": "<span style='color:#58a6ff;font-weight:bold'>S</span>",
    "E": "<span style='color:#ff6b35;font-weight:bold'>E</span>",
    "@": "<span style='color:#3fb950;font-weight:bold'>@</span>",
}


def _to_html(grid_text: str, wall_hit: bool = False) -> str:
    parts = []
    for ch in grid_text:
        if ch == "\n":
            parts.append("\n")
        elif ch == "#":
            parts.append(_WALL_HTML)
        elif ch == ".":
            parts.append(_PATH_HTML)
        elif ch in _COLORS:
            parts.append(_COLORS[ch])
        else:
            parts.append(html.escape(ch))
    inner = "".join(parts)
    style = _MAZE_STYLE_HIT if wall_hit else _MAZE_STYLE
    return f'<pre style="{style}">{inner}</pre>'


_PLACEHOLDER_HTML = _to_html(
    "#################\n"
    "#S.............E#\n"
    "#################\n"
    "\nWaiting for Phase 2 checkpoint..."
)


def _run_panel(agent_label, agent_fn, model, tokenizer, n_episodes, maze_size, max_steps):
    run_id = uuid.uuid4().hex[:12]
    seed_rng = random.Random()

    solved_count = 0
    timeout_count = 0
    steps_to_solve = []
    wall_hits_all = []
    last_grid = ""

    for ep_id in range(n_episodes):
        seed = seed_rng.randint(0, 2**31 - 1)
        ep_records = []
        ep_wall_hits = 0

        for snap in run_episode(
            agent_fn,
            maze_size=maze_size,
            max_steps=max_steps,
            seed=seed,
            episode_id=ep_id,
            run_id=run_id,
            agent_type=agent_label,
            model=model,
            tokenizer=tokenizer,
        ):
            ep_records.append(snap["step_record"])
            step = snap["step_record"]["step"]
            status = snap["step_record"]["status"]
            last_grid = snap["grid"]
            is_wall_hit = status == "wall_hit"
            if is_wall_hit:
                ep_wall_hits += 1

            status_emoji = "🧱 WALL HIT" if is_wall_hit else status
            yield (
                _to_html(snap["grid"], wall_hit=is_wall_hit),
                f"Ep {ep_id + 1}/{n_episodes}  |  Step {step}  |  {status_emoji}  |  Wall hits: {ep_wall_hits}",
                "",
            )
            time.sleep(0.3)

            if snap["done"]:
                ep_records.append(snap["episode_summary"])
                summary = snap["episode_summary"]
                if summary["solved"]:
                    solved_count += 1
                    steps_to_solve.append(summary["total_steps"])
                else:
                    timeout_count += 1
                wall_hits_all.append(summary["wall_hits"])
                break

        flush_to_dataset(ep_records, f"{run_id}_ep{ep_id:04d}", DATASET_REPO_ID)

    n = n_episodes
    solve_rate = solved_count / n * 100
    avg_steps = sum(steps_to_solve) / len(steps_to_solve) if steps_to_solve else 0
    avg_wall_hits = sum(wall_hits_all) / len(wall_hits_all) if wall_hits_all else 0
    timeout_rate = timeout_count / n * 100

    report = (
        f"Run ID: {run_id}\n"
        f"{'=' * 36}\n"
        f"Solve rate:        {solve_rate:.1f}%  ({solved_count}/{n})\n"
        f"Avg steps (wins):  {avg_steps:.1f}\n"
        f"Avg wall hits/ep:  {avg_wall_hits:.1f}\n"
        f"Timeout rate:      {timeout_rate:.1f}%  ({timeout_count}/{n})\n"
        f"Logs → {DATASET_REPO_ID}"
    )

    yield (_to_html(last_grid, wall_hit=False), "Done.", report)


def run_base(n_episodes, maze_size, max_steps):
    yield (_to_html("#################\n# Loading...    #\n#################"),
           "Loading Qwen2.5-1.5B (first run ~60s)...", "")
    try:
        model, tokenizer = load_model(BASE_MODEL)
    except Exception as e:
        yield ("", f"Model load failed: {e}", "")
        return

    yield from _run_panel(
        agent_label="qwen-1.5b-base",
        agent_fn=model_agent,
        model=model,
        tokenizer=tokenizer,
        n_episodes=int(n_episodes),
        maze_size=int(maze_size),
        max_steps=int(max_steps),
    )


def run_finetuned(n_episodes, maze_size, max_steps):
    if not _FINETUNED_AVAILABLE:
        yield (_PLACEHOLDER_HTML, "No fine-tuned model yet. Run Phase 2 first.", "")
        return

    yield ("", "Loading fine-tuned model...", "")
    try:
        model, tokenizer = load_model(FINE_TUNED_MODEL_PATH)
    except Exception as e:
        yield ("", f"Model load failed: {e}", "")
        return

    yield from _run_panel(
        agent_label="qwen-1.5b-finetuned",
        agent_fn=model_agent,
        model=model,
        tokenizer=tokenizer,
        n_episodes=int(n_episodes),
        maze_size=int(maze_size),
        max_steps=int(max_steps),
    )


# ── UI ───────────────────────────────────────────────────────────────────────

with gr.Blocks(title="knee-maze", theme=gr.themes.Base()) as demo:
    gr.Markdown("# knee-maze\nBaseline maze loop — Qwen2.5-1.5B-Instruct on CPU.")

    with gr.Row():
        # ── Base panel ───────────────────────────────────────────────────────
        with gr.Column():
            gr.Markdown("## Base (Qwen2.5-1.5B)")
            base_maze = gr.HTML(label="Maze", value=_to_html(
                "#################\n"
                "#S.............E#\n"
                "#################"
            ))
            base_status = gr.Textbox(label="Status", lines=1, interactive=False)
            with gr.Row():
                base_episodes = gr.Slider(minimum=1, maximum=50, value=5, step=1, label="Episodes")
                base_size     = gr.Slider(minimum=4, maximum=16, value=8, step=2, label="Maze size")
                base_maxsteps = gr.Slider(minimum=50, maximum=500, value=200, step=50, label="Max steps")
            base_run_btn = gr.Button("Run Base Model", variant="primary")
            base_report  = gr.Textbox(label="Report", lines=8, interactive=False)

        # ── Fine-tuned panel ─────────────────────────────────────────────────
        with gr.Column():
            gr.Markdown("## Fine-tuned")
            ft_maze = gr.HTML(label="Maze", value=_PLACEHOLDER_HTML)
            ft_status = gr.Textbox(label="Status", lines=1, interactive=False)
            with gr.Row():
                ft_episodes = gr.Slider(minimum=1, maximum=50, value=5, step=1, label="Episodes")
                ft_size     = gr.Slider(minimum=4, maximum=16, value=8, step=2, label="Maze size")
                ft_maxsteps = gr.Slider(minimum=50, maximum=500, value=200, step=50, label="Max steps")
            ft_run_btn = gr.Button(
                "Run Fine-tuned Model",
                variant="primary",
                interactive=_FINETUNED_AVAILABLE,
            )
            if not _FINETUNED_AVAILABLE:
                gr.Markdown("_No fine-tuned model yet. Run Phase 2 first._")
            ft_report = gr.Textbox(label="Report", lines=8, interactive=False)

    base_run_btn.click(
        fn=run_base,
        inputs=[base_episodes, base_size, base_maxsteps],
        outputs=[base_maze, base_status, base_report],
    )

    ft_run_btn.click(
        fn=run_finetuned,
        inputs=[ft_episodes, ft_size, ft_maxsteps],
        outputs=[ft_maze, ft_status, ft_report],
    )

demo.launch()
