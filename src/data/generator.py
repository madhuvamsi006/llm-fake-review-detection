"""
Generates Amazon-style fake reviews using GPT-4o, Claude, and Llama-3.
Reads product metadata from the Salminen dataset to ground prompts.
"""
import os
import time
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

PROMPT_TEMPLATE = (
    "Write a realistic Amazon product review for a {category} product. "
    "The review should sound like a genuine customer wrote it — specific details, "
    "natural tone, 3-5 sentences, star rating between 3 and 5. "
    "Output only the review text, no preamble."
)


def generate_gpt4o(prompts: list[str], model: str = "gpt-4o") -> list[str]:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    reviews = []
    for prompt in tqdm(prompts, desc="GPT-4o"):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.8,
            )
            reviews.append(resp.choices[0].message.content.strip())
            time.sleep(0.5)
        except Exception as e:
            print(f"GPT-4o error: {e}")
            reviews.append("")
    return reviews


def generate_claude(prompts: list[str], model: str = "claude-sonnet-4-5") -> list[str]:
    import anthropic
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    reviews = []
    for prompt in tqdm(prompts, desc="Claude"):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            reviews.append(resp.content[0].text.strip())
            time.sleep(0.5)
        except Exception as e:
            print(f"Claude error: {e}")
            reviews.append("")
    return reviews


def generate_llama3(prompts: list[str], model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct") -> list[str]:
    from transformers import pipeline
    import torch
    pipe = pipeline(
        "text-generation",
        model=model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        token=os.getenv("HF_TOKEN"),
    )
    reviews = []
    for prompt in tqdm(prompts, desc="Llama-3"):
        try:
            out = pipe(prompt, max_new_tokens=300, temperature=0.8, do_sample=True)
            text = out[0]["generated_text"].replace(prompt, "").strip()
            reviews.append(text)
        except Exception as e:
            print(f"Llama-3 error: {e}")
            reviews.append("")
    return reviews


def build_prompts(categories: list[str]) -> list[str]:
    return [PROMPT_TEMPLATE.format(category=cat) for cat in categories]


def generate_and_save(
    generator_name: str,
    categories: list[str],
    n: int = 5000,
    output_dir: str = "data/generated/",
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(output_dir) / f"{generator_name}_reviews.csv"

    if out_path.exists():
        print(f"{out_path} already exists. Skipping generation.")
        return

    import random
    sampled_cats = random.choices(categories, k=n)
    prompts = build_prompts(sampled_cats)

    if generator_name == "gpt4o":
        texts = generate_gpt4o(prompts)
    elif generator_name == "claude":
        texts = generate_claude(prompts)
    elif generator_name == "llama3":
        texts = generate_llama3(prompts)
    else:
        raise ValueError(f"Unknown generator: {generator_name}")

    df = pd.DataFrame({"text": texts, "category": sampled_cats})
    df = df[df["text"].str.strip().astype(bool)]  # drop empty
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} reviews to {out_path}")
