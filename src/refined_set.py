import json
import openai
from tqdm import tqdm
from prompt import get_fact_check_prompt

def gpt4_judge(question, gold, generation):
    prompt = get_fact_check_prompt(question, gold, generation)
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        content = response["choices"][0]["message"]["content"].strip()
        return 1 if content.startswith("1") else 0
    except Exception as e:
        print(f"GPT-4 API error: {e}")
        return -1

def refine_dataset(input_path="data/squad_final.json", output_path="data/squad_refined.json"):
    data = json.load(open(input_path))
    refined = []

    for item in tqdm(data, desc="Evaluating factuality"):
        q, gold = item["question"], item["gold"]
        for gen in item["generations"]:
            label = gpt4_judge(q, gold, gen)
            refined.append({
                "question": q,
                "gold": gold,
                "generation": gen,
                "label": label
            })

    with open(output_path, "w") as f:
        json.dump(refined, f, indent=2)
    print(f"Saved {len(refined)} labeled samples to {output_path}")
