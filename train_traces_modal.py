"""
Phase 4a: SFT on BFS reasoning traces.

Trains the base model to output step-by-step reasoning before the action.
This teaches the search algorithm, not a lookup table (Searchformer approach).

Dataset: bolajiev/knee-maze-logs/sft_traces/train.jsonl
  - Each example: structured state → reasoning trace → Action: <direction>
  - Curriculum: 30% 5×5, 40% 8×8, 30% 11×11 mazes

Output: bolajiev/qwen-maze-traces

Usage:
    modal run train_traces_modal.py
"""
import os
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4.0",
        "transformers>=4.45.0,<5.0.0",
        "trl>=1.1.0,<1.7.0",
        "peft>=0.13.0",
        "accelerate>=0.34.0",
        "datasets>=2.21.0",
        "huggingface_hub>=0.25.0",
    )
)

app = modal.App("knee-maze-traces", image=image)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

BASE_MODEL     = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET_REPO   = "bolajiev/knee-maze-logs"
DATASET_FILE   = "sft_traces/train.jsonl"
OUTPUT_REPO    = "bolajiev/qwen-maze-traces"
MAX_SEQ_LENGTH = 768
LEARNING_RATE  = 2e-4
NUM_EPOCHS     = 3
BATCH_SIZE     = 16     # A100 40GB — can push large batches
GRAD_ACCUM     = 2      # effective batch = 32


@app.function(
    gpu="A100-40GB",
    timeout=7200,
    secrets=[modal.Secret.from_name("hf-token")],
    volumes={"/root/.cache/huggingface": hf_cache},
)
def train():
    import json
    import torch
    from datasets import Dataset
    from huggingface_hub import HfApi
    from peft import LoraConfig, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    hf_token = os.environ["HF_TOKEN"]
    api = HfApi(token=hf_token)

    print("Downloading traces dataset...")
    api.hf_hub_download(
        repo_id=DATASET_REPO, filename=DATASET_FILE,
        repo_type="dataset", local_dir="/tmp/dataset", token=hf_token,
    )

    examples = []
    with open(f"/tmp/dataset/{DATASET_FILE}") as f:
        for line in f:
            examples.append(json.loads(line))
    print(f"Loaded {len(examples)} examples (using all)")
    dataset = Dataset.from_list(examples)

    print(f"Loading {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = MAX_SEQ_LENGTH
    tokenizer.truncation_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map={"": 0}, token=hf_token,
    )

    lora_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM,
    )

    def format_example(example):
        return tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False,
        )

    training_args = SFTConfig(
        output_dir="/tmp/traces-output",
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=50,
        save_strategy="epoch",
        dataset_text_field=None,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=lora_config,
        formatting_func=format_example,
    )

    print("Training...")
    trainer.train()

    print("Merging LoRA weights...")
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained("/tmp/traces-merged")
    tokenizer.save_pretrained("/tmp/traces-merged")

    print(f"Pushing to {OUTPUT_REPO}...")
    api.create_repo(OUTPUT_REPO, repo_type="model", private=True, exist_ok=True, token=hf_token)
    api.upload_folder(
        folder_path="/tmp/traces-merged",
        repo_id=OUTPUT_REPO, repo_type="model", token=hf_token,
    )
    print(f"Done — huggingface.co/{OUTPUT_REPO}")


@app.local_entrypoint()
def main():
    train.remote()
