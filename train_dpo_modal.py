"""
Phase 3: DPO fine-tuning on top of the Phase 2 SFT checkpoint.

Trains on preference pairs: chosen=BFS-optimal direction, rejected=worst valid direction.
Starts from bolajiev/qwen-maze-sft (not the base model).

Usage:
    modal run train_dpo_modal.py
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

app = modal.App("knee-maze-dpo", image=image)

hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

SFT_MODEL       = "bolajiev/qwen-maze-sft"
DATASET_REPO    = "bolajiev/knee-maze-logs"
DATASET_FILE    = "dpo/train.jsonl"
OUTPUT_REPO     = "bolajiev/qwen-maze-dpo"
BETA            = 0.1
LEARNING_RATE   = 5e-5
NUM_EPOCHS      = 1
BATCH_SIZE      = 4
GRAD_ACCUM      = 4


@app.function(
    gpu="T4",
    timeout=10800,
    secrets=[modal.Secret.from_name("hf-token")],
    volumes={"/root/.cache/huggingface": hf_cache},
)
def train():
    import json
    import tempfile

    import torch
    from datasets import Dataset
    from huggingface_hub import HfApi
    from peft import LoraConfig, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    hf_token = os.environ["HF_TOKEN"]
    api = HfApi(token=hf_token)

    print("Downloading DPO dataset...")
    api.hf_hub_download(
        repo_id=DATASET_REPO,
        filename=DATASET_FILE,
        repo_type="dataset",
        local_dir="/tmp/dataset",
        token=hf_token,
    )

    examples = []
    with open(f"/tmp/dataset/{DATASET_FILE}") as f:
        for line in f:
            examples.append(json.loads(line))
    print(f"Loaded {len(examples)} DPO pairs, using first 15000")
    examples = examples[:15000]
    dataset = Dataset.from_list(examples)

    print(f"Loading {SFT_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = 512
    tokenizer.truncation_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        SFT_MODEL,
        torch_dtype=torch.float16,
        device_map={"": 0},
        token=hf_token,
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    training_args = DPOConfig(
        output_dir="/tmp/dpo-output",
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        fp16=True,
        logging_steps=50,
        save_strategy="epoch",
        beta=BETA,
        report_to="none",
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    print("Training...")
    trainer.train()

    print("Merging LoRA weights...")
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained("/tmp/dpo-merged")
    tokenizer.save_pretrained("/tmp/dpo-merged")

    print(f"Pushing to {OUTPUT_REPO}...")
    api.create_repo(OUTPUT_REPO, repo_type="model", private=True, exist_ok=True, token=hf_token)
    api.upload_folder(
        folder_path="/tmp/dpo-merged",
        repo_id=OUTPUT_REPO,
        repo_type="model",
        token=hf_token,
    )
    print(f"Done — checkpoint at huggingface.co/{OUTPUT_REPO}")


@app.local_entrypoint()
def main():
    train.remote()
