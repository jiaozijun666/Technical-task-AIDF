import os
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODELS = Path(__file__).resolve().parents[1] / "hf_models"
MODELS.mkdir(exist_ok=True)

def load_llm(model_name: str, eightbit: bool = True, flash: bool = True):
    if flash:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    quant = BitsAndBytesConfig(load_in_8bit=True) if eightbit else None
    tok = AutoTokenizer.from_pretrained(model_name, cache_dir=str(MODELS), use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    kwargs = dict(torch_dtype=torch.float16, device_map="auto")
    if quant:
        kwargs["quantization_config"] = quant
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=str(MODELS), **kwargs)
    model.eval()
    return tok, model
