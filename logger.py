import json
import os
import tempfile


def flush_to_dataset(records: list, run_id: str, repo_id: str) -> bool:
    """
    Write records as JSONL and upload to a private HF Dataset repo.
    Returns True on success, False on failure (logs the error, never raises).
    Requires HF_TOKEN env var with write access to the dataset repo.
    """
    hf_token = os.getenv("HF_TOKEN", "")
    if not hf_token:
        print(f"[logger] HF_TOKEN not set — skipping dataset upload for run {run_id}")
        return False
    if not repo_id or repo_id.startswith("<"):
        print(f"[logger] DATASET_REPO_ID not configured — skipping upload for run {run_id}")
        return False

    try:
        from huggingface_hub import HfApi

        jsonl = "\n".join(json.dumps(r) for r in records) + "\n"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(jsonl)
            tmp_path = f.name

        api = HfApi(token=hf_token)
        api.upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo=f"logs/{run_id}.jsonl",
            repo_id=repo_id,
            repo_type="dataset",
        )
        os.remove(tmp_path)
        print(f"[logger] Uploaded {len(records)} records → {repo_id}/logs/{run_id}.jsonl")
        return True

    except Exception as e:
        print(f"[logger] Upload failed for run {run_id}: {e}")
        return False
