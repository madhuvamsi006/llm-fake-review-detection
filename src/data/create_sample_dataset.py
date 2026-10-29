"""
Creates a synthetic sample dataset matching the Salminen et al. (2022) structure.
Used when the OSF dataset is unavailable for download.

To use the real dataset: download from https://osf.io/tyue9/
and save as data/raw/fake_reviews_dataset.csv
"""
import random
import pandas as pd
from pathlib import Path

random.seed(42)

CATEGORIES = [
    "Electronics", "Books", "Clothing", "Sports", "Home & Kitchen",
    "Beauty", "Toys", "Automotive", "Health", "Garden",
    "Office Products", "Pet Supplies", "Music", "Movies", "Food",
]

GENUINE_TEMPLATES = [
    "I bought this {cat} product and it works exactly as described. Very happy with the quality.",
    "This is my second purchase of this item. Still great after {n} months of use.",
    "Decent product for the price. Does what it says on the tin. Would recommend.",
    "Great {cat} product. Fast delivery and well packaged. Works perfectly.",
    "I was skeptical at first but this exceeded my expectations. Good build quality.",
    "Solid product. Nothing fancy but does the job well. Good value for money.",
    "Bought this as a gift and the recipient loved it. Will buy again.",
    "Not bad. A couple of minor issues but overall satisfied with the purchase.",
    "Arrived quickly and in perfect condition. Exactly what I needed for my {cat} project.",
    "Very impressed with this. Easy to set up and works flawlessly. Highly recommend.",
    "Works as advertised. Good quality for the price point.",
    "Had this for {n} weeks now and no complaints. Does exactly what it's supposed to.",
    "Excellent product. The quality is much better than I expected at this price.",
    "Good product overall. Packaging was secure and delivery was fast.",
    "Does the job. Nothing spectacular but solid and reliable.",
]

GPT2_TEMPLATES = [
    "This product is a great product. The product works well and the product quality is good product.",
    "I love this product. This product is the best product. Product is very product.",
    "The {cat} is very good. I like the {cat}. The {cat} is great for {cat} things.",
    "Great great great. Very very very good product product product. Buy buy buy this product.",
    "This is the best product I have ever used in my entire life for all my needs and requirements.",
    "Product works. Product is good. Product arrived. Product was in box. Product is product.",
    "I purchased this item and the item is an item that does item things with item quality.",
    "Very satisfied customer here. This product exceeded all expectation of product excellence.",
    "The quality of this product quality is of high quality and the quality is quality assured.",
    "Recommend this product to everyone everywhere always for all purposes and all situations.",
    "Five stars. Five stars. Five stars product. Stars are five. Product gets five of the stars.",
    "This item item item is good good good for doing the thing thing thing it does does does.",
    "Best purchase ever made in history of purchases. Will purchase again and again forever.",
    "Product description matches product. Product arrived in product packaging. Good product.",
    "I am very happy with this purchase and will tell all my friends to purchase this purchase.",
]


def generate_review(template: str, cat: str) -> str:
    n = random.randint(2, 12)
    return template.format(cat=cat.lower(), n=n)


def create_sample_dataset(
    n_genuine: int = 5000,
    n_gpt2: int = 5000,
    out_path: str = "data/raw/fake_reviews_dataset.csv",
):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for _ in range(n_genuine):
        cat = random.choice(CATEGORIES)
        tmpl = random.choice(GENUINE_TEMPLATES)
        text = generate_review(tmpl, cat)
        rating = random.choices([4, 5, 3, 2, 1], weights=[40, 35, 15, 7, 3])[0]
        rows.append({"category": cat, "rating": rating, "label": "OR", "text_": text})

    for _ in range(n_gpt2):
        cat = random.choice(CATEGORIES)
        tmpl = random.choice(GPT2_TEMPLATES)
        text = generate_review(tmpl, cat)
        rating = random.choices([5, 4, 3], weights=[60, 30, 10])[0]
        rows.append({"category": cat, "rating": rating, "label": "CG", "text_": text})

    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(out_path, index=False)
    print(f"Sample dataset saved: {out_path}")
    print(df["label"].value_counts().to_string())
    return df


if __name__ == "__main__":
    create_sample_dataset()
