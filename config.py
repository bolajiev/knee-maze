import os

from dotenv import load_dotenv

load_dotenv()

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
FINE_TUNED_MODEL_PATH = "bolajiev/qwen-maze-dpo"

DATASET_REPO_ID: str = "bolajiev/knee-maze-logs"

DEFAULTS = {
    "maze_size": 8,
    "max_steps": 200,
    "episodes": 5,
    "temperature": 0.0,
}
