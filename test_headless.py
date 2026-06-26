"""
Headless sanity check — no Gradio, no HF Hub, no model weights needed.
Just runs random_agent for N episodes through runner.py and verifies:
  - no crashes
  - all step records are valid dicts with required keys
  - episode summaries are correct
  - maze is deterministic per seed
"""
import json
import uuid

from agent import random_agent
from maze import generate_maze, render
from runner import run_episode

REQUIRED_STEP_KEYS = {
    "record_type", "run_id", "episode_id", "step", "maze_seed",
    "agent_type", "position_before", "action", "valid_move",
    "position_after", "status", "raw_model_output", "timestamp",
}
REQUIRED_SUMMARY_KEYS = {
    "record_type", "run_id", "episode_id", "maze_seed", "agent_type",
    "maze_size", "solved", "total_steps", "wall_hits", "reason_ended",
}

N_EPISODES = 20
MAZE_SIZE = 8
MAX_STEPS = 200


def test_determinism():
    m1 = generate_maze(MAZE_SIZE, seed=9999)
    m2 = generate_maze(MAZE_SIZE, seed=9999)
    assert m1.passages == m2.passages, "maze not deterministic"
    m3 = generate_maze(MAZE_SIZE, seed=1234)
    assert m1.passages != m3.passages, "different seeds produced same maze"
    print("  determinism: OK")


def test_episodes():
    run_id = uuid.uuid4().hex[:12]
    solved = 0
    timeouts = 0
    wall_hits_total = 0
    steps_to_solve = []

    import random as _random
    seed_rng = _random.Random(42)

    for ep_id in range(N_EPISODES):
        seed = seed_rng.randint(0, 2**31 - 1)
        step_count = 0
        ep_summary = None

        for snap in run_episode(
            random_agent,
            maze_size=MAZE_SIZE,
            max_steps=MAX_STEPS,
            seed=seed,
            episode_id=ep_id,
            run_id=run_id,
            agent_type="random",
        ):
            rec = snap["step_record"]

            # Validate required keys
            missing = REQUIRED_STEP_KEYS - set(rec.keys())
            assert not missing, f"step record missing keys: {missing}"

            # Round-trip through JSON (catches non-serialisable values)
            json.loads(json.dumps(rec))

            step_count += 1

            if snap["done"]:
                assert snap["episode_summary"] is not None
                ep_summary = snap["episode_summary"]
                missing_s = REQUIRED_SUMMARY_KEYS - set(ep_summary.keys())
                assert not missing_s, f"summary missing keys: {missing_s}"
                json.loads(json.dumps(ep_summary))

        assert ep_summary is not None, f"episode {ep_id} never yielded a summary"
        assert ep_summary["total_steps"] == step_count

        if ep_summary["solved"]:
            solved += 1
            steps_to_solve.append(ep_summary["total_steps"])
        else:
            timeouts += 1
        wall_hits_total += ep_summary["wall_hits"]

    n = N_EPISODES
    print(f"  episodes:      {n}")
    print(f"  solve rate:    {solved/n*100:.1f}%  ({solved}/{n})")
    print(f"  timeout rate:  {timeouts/n*100:.1f}%")
    print(f"  avg wall hits: {wall_hits_total/n:.1f}")
    if steps_to_solve:
        print(f"  avg steps/win: {sum(steps_to_solve)/len(steps_to_solve):.1f}")
    print("  all records valid JSON: OK")


if __name__ == "__main__":
    print(f"headless test — random_agent, {N_EPISODES} episodes, {MAZE_SIZE}x{MAZE_SIZE}, max {MAX_STEPS} steps\n")
    test_determinism()
    test_episodes()
    print("\nall checks passed.")
