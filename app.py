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
    "font-size:15px;"
    "line-height:1.35;"
    "background:#0d1117;"
    "padding:14px 16px;"
    "border-radius:8px;"
    "margin:0;"
    "display:inline-block;"
    "min-width:100%;"
)

_COLORS = {
    "#": ("<span style='color:#3a3a4a'>", "</span>"),
    ".": ("<span style='color:#2a2a3a'>", "</span>"),
    "S": ("<span style='color:#58a6ff;font-weight:bold'>", "</span>"),
    "E": ("<span style='color:#ff6b35;font-weight:bold'>", "</span>"),
    "@": ("<span style='color:#3fb950;font-weight:bold;font-size:17px'>", "</span>"),
}


def _to_html(grid_text: str) -> str:
    parts = []
    for ch in grid_text:
        if ch == "\n":
            parts.append("\n")
        elif ch in _COLORS:
            open_tag, close_tag = _COLORS[ch]
            parts.append(f"{open_tag}{ch}{close_tag}")
        else:
            parts.append(html.escape(ch))
    inner = "".join(parts)
    return f'<pre style="{_MAZE_STYLE}">{inner}</pre>'


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

            yield (
                _to_html(snap["grid"]),
                f"Episode {ep_id + 1}/{n_episodes}  |  Step {step}  |  {status}",
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

    yield (_to_html(last_grid), "Done.", report)


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
