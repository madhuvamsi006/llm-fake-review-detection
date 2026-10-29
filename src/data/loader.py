"""
Loads and merges all review datasets into a unified DataFrame.

Columns in unified dataset:
    text       : review text
    label      : 0 = genuine, 1 = LLM-generated
    generator  : genuine | gpt2 | gpt4o | claude | llama3
    category   : product category (if available)
"""
import pandas as pd
from pathlib import Path


def load_salminen(path: str = "data/raw/fake_reviews_dataset.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    # Salminen dataset columns: 'category', 'rating', 'label', 'text_'
    df = df.rename(columns={"text_": "text"})
    df["label"] = df["label"].map({"OR": 0, "CG": 1})   # OR=Original/Genuine, CG=Computer-Generated
    df["generator"] = df["label"].map({0: "genuine", 1: "gpt2"})
    return df[["text", "label", "generator", "category"]].dropna(subset=["text"])


def load_generated(generator: str, path: str = None) -> pd.DataFrame:
    if path is None:
        path = f"data/generated/{generator}_reviews.csv"
    df = pd.read_csv(path)
    df["label"] = 1
    df["generator"] = generator
    if "category" not in df.columns:
        df["category"] = "unknown"
    return df[["text", "label", "generator", "category"]].dropna(subset=["text"])


def load_all(
    salminen_path: str = "data/raw/fake_reviews_dataset.csv",
    generated_generators: list = None,
) -> pd.DataFrame:
    if generated_generators is None:
        generated_generators = ["gpt4o", "claude", "llama3"]

    frames = [load_salminen(salminen_path)]

    for gen in generated_generators:
        gen_path = f"data/generated/{gen}_reviews.csv"
        if Path(gen_path).exists():
            frames.append(load_generated(gen, gen_path))
        else:
            print(f"[WARNING] Generated reviews not found: {gen_path}. Skipping.")

    df = pd.concat(frames, ignore_index=True)
    df = df.reset_index(drop=True)
    print(f"Loaded {len(df):,} reviews | label dist:\n{df['label'].value_counts().to_string()}")
    print(f"Generator dist:\n{df['generator'].value_counts().to_string()}")
    return df
