import functools

from transformers import AutoModelForCausalLM, AutoTokenizer


@functools.lru_cache(maxsize=2)
def load_model(model_path: str):
    """Load and cache (model, tokenizer). Raises on bad path — caller handles."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tokenizer
