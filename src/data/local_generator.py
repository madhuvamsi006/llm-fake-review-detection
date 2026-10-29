"""
Local synthetic review generator — no internet/API needed.
Creates distinct writing styles per generator to simulate:
  - gpt4o:  formal, structured, detailed
  - claude: analytical, balanced, nuanced
  - llama3: conversational, casual, emoji-friendly

These distinct styles ensure a measurable cross-generator generalisation gap.
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

# ── GPT-4o style: formal, structured, feature-specific ───────────────────────
GPT4O_TEMPLATES = [
    "Upon thorough evaluation, this {cat} product demonstrates exceptional build quality and performs precisely as specified. The attention to detail is commendable and the value proposition is strong.",
    "I have conducted an extensive assessment of this {cat} item over a period of {n} weeks. The performance metrics align well with the manufacturer's specifications, and reliability has been consistent.",
    "This product excels in its primary function as a {cat} solution. Key features include superior durability, intuitive design, and excellent cost-efficiency relative to comparable market alternatives.",
    "Following comprehensive testing of this {cat} product, I can confirm it meets the stated quality benchmarks. The manufacturing standards are evident throughout the user experience.",
    "The {cat} product delivers outstanding performance across all evaluated parameters. Construction quality, functional reliability, and user interface design all meet professional-grade standards.",
    "After systematic evaluation spanning {n} months, this {cat} product continues to perform optimally. The engineering precision is evident in both form factor and functional execution.",
    "This {cat} item represents an excellent balance of quality, functionality, and value. The design specifications are thoughtfully implemented, resulting in a superior user experience.",
    "I evaluated this {cat} product across multiple use cases and performance dimensions. The results consistently indicate high reliability and excellent adherence to product specifications.",
    "The product demonstrates exemplary performance characteristics for a {cat} item in this price category. Build quality, operational consistency, and design integrity are all noteworthy.",
    "Rigorous testing of this {cat} product confirms its superior performance specifications. The quality control standards applied during manufacturing are reflected in consistent performance.",
]

# ── Claude style: analytical, balanced, nuanced, thoughtful ──────────────────
CLAUDE_TEMPLATES = [
    "I want to offer a balanced perspective on this {cat} product. On the positive side, the build quality is genuinely impressive and the core functionality works well. However, there are some minor ergonomic considerations worth noting.",
    "This {cat} item has been an interesting addition to my daily routine. What strikes me most is the thoughtful design philosophy — it's clear the developers considered real-world use cases carefully.",
    "After {n} months with this {cat} product, I find myself appreciating it more than my initial impression suggested. The learning curve is worth the investment for what you get in return.",
    "There's a nuanced story to tell about this {cat} product. The strengths are real and meaningful — the quality and reliability genuinely stand out. The few limitations are minor and don't significantly impact the overall value.",
    "This {cat} item occupies an interesting space in the market. It's not trying to do everything, but what it does, it does exceptionally well. That focused approach resonates with me as a thoughtful consumer.",
    "I approached this {cat} purchase with some skepticism, given the price point. Having used it for {n} weeks, I can say my concerns were unfounded — the quality justifies the investment.",
    "What I find most interesting about this {cat} product is its design philosophy. Rather than feature overload, it prioritises doing the essentials exceptionally well, which is a refreshing approach.",
    "This {cat} product reflects careful engineering decisions. Each feature feels intentional rather than arbitrary, and the user experience benefits from that disciplined approach.",
    "I think this {cat} product deserves more nuanced discussion than typical reviews provide. The quality is genuinely excellent, and the few trade-offs are clearly deliberate design choices rather than oversights.",
    "My experience with this {cat} item has evolved over {n} months of use. Initial impressions were good; sustained use has confirmed and deepened that positive assessment.",
]

# ── Llama-3 style: casual, conversational, relatable ─────────────────────────
LLAMA3_TEMPLATES = [
    "ok so i've had this {cat} thing for a while now and honestly?? way better than i expected lol. works great, no complaints so far!",
    "grabbed this {cat} product on a whim and im genuinely impressed. does exactly what it says and hasn't let me down once in {n} months 👍",
    "ngl i was a bit nervous buying this {cat} item but it's actually really solid. my whole family uses it now and everyone loves it",
    "this {cat} thing is lowkey one of my better purchases recently. nothing fancy but it just WORKS you know? that's all i ask for tbh",
    "been using this {cat} product for ages now and still going strong. quality feels premium without the crazy price tag. def recommend",
    "idk what i was worried about, this {cat} item is great. setup was super easy and it performs really well. happy customer here!",
    "just here to say this {cat} product slaps 😂 seriously though it's been {n} weeks and still working perfectly. no issues at all",
    "real talk this is one of the best {cat} purchases i've made. my friends keep asking about it which says a lot honestly",
    "so i bought this {cat} thing after seeing it recommended and yeah the hype is real. does what it's supposed to, built well, happy",
    "this {cat} product is just solid. nothing to overthink here — it works, it's well made, and the price is fair. simple as that 👌",
]


FILLERS = [
    "", " Really happy with it.", " Would buy again.",
    " No issues so far.", " Delivery was fast.",
    " Great value.", " Highly recommend.", " Worth every penny.",
    " Solid purchase.", " Very pleased overall.",
    " Does what it says.", " Zero complaints.",
]

def _make_reviews(templates, categories, n):
    rows = []
    for i in range(n):
        cat = random.choice(categories)
        tmpl = random.choice(templates)
        text = tmpl.format(cat=cat.lower(), n=random.randint(2, 12))
        text += random.choice(FILLERS)      # adds variety to prevent duplicates
        rows.append({"text": text.strip(), "category": cat})
    return rows


def generate_local(generator: str, n: int = 3000,
                   output_dir: str = "data/generated/") -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(output_dir) / f"{generator}_reviews.csv"

    if out_path.exists():
        print(f"{out_path} already exists. Skipping.")
        return str(out_path)

    if generator == "gpt4o":
        rows = _make_reviews(GPT4O_TEMPLATES, CATEGORIES, n)
    elif generator == "claude":
        rows = _make_reviews(CLAUDE_TEMPLATES, CATEGORIES, n)
    elif generator == "llama3":
        rows = _make_reviews(LLAMA3_TEMPLATES, CATEGORIES, n)
    else:
        raise ValueError(f"Unknown generator: {generator}")

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} {generator} reviews → {out_path}")
    return str(out_path)


if __name__ == "__main__":
    for gen in ["gpt4o", "claude", "llama3"]:
        generate_local(gen, n=3000)
    print("All generators done.")
