"""
Phase 2: LoRA SFT fine-tuning on BFS-optimal maze trajectories.
Runs on Modal (T4 GPU). Pushes checkpoint to HF Hub.

Usage:
    modal run train_modal.py
    modal run train_modal.py --detach   # fire and forget, check logs later
"""
import os

import modal

# ── Modal image ──────────────────────────────────────────────────────────────

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.0",
        "transformers>=4.45.0",
        "trl>=0.11.0",
        "peft>=0.13.0",
        "accelerate>=0.34.0",
        "datasets>=2.21.0",
        "huggingface_hub>=0.25.0",
        "bitsandbytes>=0.43.0",
    )
)

app = modal.App("knee-maze-sft", image=image)

# Persist the HF cache across runs so we don't re-download the base model
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

# ── Config ────────────────────────────────────────────────────────────────────

BASE_MODEL      = "Qwen/Qwen2.5-1.5B-Instruct"
DATASET_REPO    = "bolajiev/knee-maze-logs"
DATASET_FILE    = "sft/train.jsonl"
OUTPUT_REPO     = "bolajiev/qwen-maze-sft"
MAX_SEQ_LENGTH  = 512
LORA_R          = 16
LORA_ALPHA      = 32
LORA_DROPOUT    = 0.05
LEARNING_RATE   = 2e-4
NUM_EPOCHS      = 3
BATCH_SIZE      = 8
GRAD_ACCUM      = 4   # effective batch = 32


# ── Training function ─────────────────────────────────────────────────────────

@app.function(
    gpu="T4",
    timeout=7200,          # 2 hours max
    secrets=[modal.Secret.from_name("hf-token")],
    volumes={"/root/.cache/huggingface": hf_cache},
)
def train():
    import json
    import tempfile

    import torch
    from datasets import Dataset
    from huggingface_hub import HfApi, snapshot_download
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from trl import SFTConfig, SFTTrainer

    hf_token = os.environ["HF_TOKEN"]

    # ── Load dataset ──────────────────────────────────────────────────────────
    print("Downloading dataset...")
    api = HfApi(token=hf_token)

    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        tmp_path = f.name

    api.hf_hub_download(
        repo_id=DATASET_REPO,
        filename=DATASET_FILE,
        repo_type="dataset",
        local_dir="/tmp/dataset",
        token=hf_token,
    )
    dataset_path = f"/tmp/dataset/{DATASET_FILE}"

    examples = []
    with open(dataset_path) as f:
        for line in f:
            examples.append(json.loads(line))

    print(f"Loaded {len(examples)} training examples")
    dataset = Dataset.from_list(examples)

    # ── Load model + tokenizer ────────────────────────────────────────────────
    print(f"Loading {BASE_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=hf_token,
    )

    # ── LoRA ──────────────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    def format_example(example):
        return tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )

    training_args = SFTConfig(
        output_dir="/tmp/sft-output",
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=50,
        save_strategy="epoch",
        max_seq_length=MAX_SEQ_LENGTH,
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

    # ── Merge LoRA + push ─────────────────────────────────────────────────────
    print("Merging LoRA weights...")
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained("/tmp/sft-merged")
    tokenizer.save_pretrained("/tmp/sft-merged")

    print(f"Pushing to {OUTPUT_REPO}...")
    api.create_repo(OUTPUT_REPO, repo_type="model", private=True, exist_ok=True, token=hf_token)
    api.upload_folder(
        folder_path="/tmp/sft-merged",
        repo_id=OUTPUT_REPO,
        repo_type="model",
        token=hf_token,
    )
    print(f"Done — checkpoint at huggingface.co/{OUTPUT_REPO}")


# ── Local entrypoint ──────────────────────────────────────────────────────────

@app.local_entrypoint()
def main():
    train.remote()
