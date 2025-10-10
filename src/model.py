from __future__ import annotations
import os
import torch
from dataclasses import dataclass
from typing import Optional, Union, Literal, Any
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments
)
import requests

@dataclass
class GenConfig:
    temperature: float = 0.5
    top_p: float = 1.0
    top_k: int = 50
    max_new_tokens: int = 64
    do_sample: bool = True


@dataclass
class GenOutput:
    text: str
    raw: Any = None

class BaseModelClient:
    backend: str
    def generate(self, prompt: str, cfg: GenConfig) -> GenOutput:
        raise NotImplementedError

    def nll(self, prompt: str, answer: str) -> float:
        """Compute negative log-likelihood for uncertainty baselines."""
        raise NotImplementedError


class HFModelClient(BaseModelClient):
    def __init__(self, model_id: str, device: Optional[str] = None):
        self.backend = "hf"
        self.model_id = model_id
        self.device = self._resolve_device(device)

        print(f"[INFO] Loading model: {model_id} on {self.device}...")
        self.tok = AutoTokenizer.from_pretrained(model_id)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

        self.mdl = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
        ).to(self.device)
        self.mdl.eval()

    def generate(self, prompt: str, cfg: GenConfig) -> GenOutput:
        enc = self.tok(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.mdl.generate(**enc, **cfg.__dict__, pad_token_id=self.tok.pad_token_id)
        gen_text = self.tok.decode(out[0], skip_special_tokens=True)
        return GenOutput(text=gen_text)

    def nll(self, prompt: str, answer: str) -> float:
        """Return negative log-likelihood of model generating answer given prompt."""
        text = f"{prompt}\n{answer}"
        enc = self.tok(text, return_tensors="pt").to(self.device)
        input_ids = enc["input_ids"]
        with torch.no_grad():
            out = self.mdl(input_ids, labels=input_ids)
            nll = out.loss.item()
        return -nll  # higher = more confident

    def _resolve_device(self, device: Optional[str]):
        if device:
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

class OllamaClient(BaseModelClient):
    def __init__(self, model_id: str, api_base: str = "http://127.0.0.1:11434"):
        self.backend = "ollama"
        self.model_id = model_id
        self.api = api_base

    def generate(self, prompt: str, cfg: GenConfig) -> GenOutput:
        res = requests.post(
            f"{self.api}/api/generate",
            json={"model": self.model_id, "prompt": prompt, "stream": False},
            timeout=600
        )
        res.raise_for_status()
        text = res.json().get("response", "").strip()
        return GenOutput(text=text)


def finetune_model(base_model_id: str, dataset_path: str, output_dir: str, epochs: int = 1):
    """
    Fine-tunes a causal LM on a JSON dataset with keys 'prompt' and 'response'.
    """
    print(f"[INFO] Loading base model: {base_model_id}")

    # --- Ensure dataset path works in any environment ---
    if not os.path.exists(dataset_path):
        alt_path = os.path.join("/workspace/Technical-task-AIDF", dataset_path)
        if os.path.exists(alt_path):
            dataset_path = alt_path
        else:
            raise FileNotFoundError(f"Dataset not found at {dataset_path} or {alt_path}")

    print(f"[INFO] Loading dataset from {dataset_path}")
    dataset = load_dataset("json", data_files=dataset_path)
    if "train" in dataset:
        dataset = dataset["train"]

    print(f"[INFO] Dataset loaded successfully ({len(dataset)} samples).")

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_fn(batch):
        return tokenizer(
            batch["prompt"],
            text_target=batch["response"],
            truncation=True,
            padding="max_length",
            max_length=128
        )

    tokenized = dataset.map(tokenize_fn, batched=True)
    model = AutoModelForCausalLM.from_pretrained(base_model_id)

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        num_train_epochs=epochs,
        learning_rate=5e-5,
        save_strategy="epoch",
        logging_steps=20,
        fp16=torch.cuda.is_available(),
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        tokenizer=tokenizer
    )

    print("[INFO] Starting fine-tuning...")
    trainer.train()
    trainer.save_model(output_dir)
    print(f"[INFO] Fine-tuning complete. Model saved to {output_dir}")


def get_model(model_id: str, backend: Literal["hf", "ollama"] = "hf") -> BaseModelClient:
    if backend == "hf":
        return HFModelClient(model_id)
    elif backend == "ollama":
        return OllamaClient(model_id)
    else:
        raise ValueError(f"Unknown backend: {backend}")
