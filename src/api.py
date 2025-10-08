import os
from src.model import get_model, GenConfig

def run_inference(prompt: str, model_id="meta-llama/Llama-3.1-8B-Instruct", backend="hf"):
    """
    Unified API for all model calls.
    Handles both local HF and OpenAI API models.
    """
    if backend == "hf":
        model = get_model(model_id, backend="hf")
        cfg = GenConfig(temperature=0.5, top_p=0.9, max_new_tokens=64)
        return model.generate(prompt, cfg).text

    elif backend == "openai":
        import openai
        openai.api_key = os.getenv("OPENAI_API_KEY")
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response["choices"][0]["message"]["content"].strip()

    else:
        raise ValueError(f"Unsupported backend: {backend}")
