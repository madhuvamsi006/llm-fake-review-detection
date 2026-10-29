"""
Full end-to-end pipeline:
  load → split → features → XGBoost → evaluate (in-dist + cross-gen) → cost → results

Run: venv/bin/python3 run_pipeline.py
"""
import json, time
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.sparse import hstack, csr_matrix
import nltk
for r in ["punkt", "punkt_tab", "averaged_perceptron_tagger",
          "averaged_perceptron_tagger_eng", "stopwords"]:
    nltk.download(r, quiet=True)

from src.data.loader import load_all
from src.data.splitter import product_aware_split
from src.features.classical import fit_transform_classical
from src.models.xgboost_model import train_xgboost, save_model
from src.evaluation.metrics import evaluate, cross_gen_drop, results_to_dataframe, mcnemar_test
from src.evaluation.cost import benchmark_xgboost, cost_table

for d in ["results/metrics","results/figures","results/cost_analysis","results/checkpoints"]:
    Path(d).mkdir(parents=True, exist_ok=True)

print("\n" + "="*60)
print("  LLM Fake Review Detection — Full Pipeline")
print("="*60)

# ── 1. Load all data ──────────────────────────────────────────
print("\n[1/7] Loading dataset (genuine + gpt2 + gpt4o + claude + llama3)...")
df = load_all(generated_generators=["gpt4o", "claude", "llama3"])
print(f"\nTotal: {len(df):,} reviews")
print(df["generator"].value_counts().to_string())

# ── 2. In-distribution split (train on all generators) ───────
print("\n[2/7] Leakage-controlled splitting (in-distribution)...")
train_id, val_id, test_id = product_aware_split(df, random_seed=42)

# ── 3. Cross-generator split (hold out llama3) ───────────────
print("\n[3/7] Cross-generator split (hold out llama3)...")
# Train on genuine + gpt2 + gpt4o + claude; test on llama3
held_out = "llama3"
train_cg = df[df["generator"] != held_out].copy()
test_cg  = df[df["generator"] == held_out].copy()

# Product-aware split for cross-gen training set
train_cg_split, val_cg_split, _ = product_aware_split(train_cg, random_seed=42)
print(f"Cross-gen train: {len(train_cg_split):,} | test (llama3): {len(test_cg):,}")

# ── 4. Feature extraction ─────────────────────────────────────
print("\n[4/7] Extracting features...")

# In-distribution features
feat_id = fit_transform_classical(
    train_id["text"].tolist(),
    val_id["text"].tolist(),
    test_id["text"].tolist(),
)
X_train_id = feat_id["train"]
X_test_id  = feat_id["test"]
tfidf_id   = feat_id["tfidf"]
stylo_id   = feat_id["stylo"]
print(f"In-dist feature shape: {X_train_id.shape}")

# Cross-generator features
feat_cg = fit_transform_classical(
    train_cg_split["text"].tolist(),
    val_cg_split["text"].tolist(),
)
X_train_cg = feat_cg["train"]
tfidf_cg   = feat_cg["tfidf"]
stylo_cg   = feat_cg["stylo"]
X_test_cg  = hstack([
    tfidf_cg.transform(test_cg["text"].tolist()),
    csr_matrix(stylo_cg.transform(test_cg["text"].tolist()))
])
print(f"Cross-gen feature shape: {X_train_cg.shape}")

# ── 5. Train XGBoost (both conditions) ───────────────────────
print("\n[5/7] Training XGBoost...")
t0 = time.time()
model_id = train_xgboost(X_train_id, train_id["label"].values)
save_model(model_id, "results/checkpoints/xgboost_indist.joblib")

model_cg = train_xgboost(X_train_cg, train_cg_split["label"].values)
save_model(model_cg, "results/checkpoints/xgboost_crossgen.joblib")
train_time = time.time() - t0
print(f"Training time: {train_time:.1f}s")

# ── 6. Evaluate ───────────────────────────────────────────────
print("\n[6/7] Evaluating...")

# In-distribution
y_pred_id  = model_id.predict(X_test_id)
y_prob_id  = model_id.predict_proba(X_test_id)[:, 1]
res_id     = evaluate(test_id["label"].values, y_pred_id, y_prob_id,
                      label="XGBoost (In-Distribution)")

# Cross-generator
y_pred_cg  = model_cg.predict(X_test_cg)
y_prob_cg  = model_cg.predict_proba(X_test_cg)[:, 1]
res_cg     = evaluate(test_cg["label"].values, y_pred_cg, y_prob_cg,
                      label="XGBoost (Cross-Generator: unseen Llama-3)")

drop = cross_gen_drop(res_id["f1"], res_cg["f1"])

# ── 7. Cost benchmarking ──────────────────────────────────────
print("\n[7/7] Cost benchmarking...")
cost = benchmark_xgboost(model_id, X_test_id, n_runs=5)
cost["training_time_s"] = round(train_time, 2)
df_cost = cost_table([cost])

# ── Save results ──────────────────────────────────────────────
df_metrics = results_to_dataframe([res_id, res_cg])
df_metrics.to_csv("results/metrics/xgboost_results.csv", index=False)

summary = {
    "in_distribution": {k: round(v,4) for k,v in res_id.items() if isinstance(v, float)},
    "cross_generator":  {k: round(v,4) for k,v in res_cg.items() if isinstance(v, float)},
    "cross_gen_drop":  round(drop, 4),
    "held_out_generator": held_out,
    "training_time_s": round(train_time, 2),
    "dataset_size": {
        "total": len(df),
        "genuine": int((df["label"]==0).sum()),
        "llm_generated": int((df["label"]==1).sum()),
    },
    "inference_cost": {
        "latency_ms_per_review": cost["Latency/review (ms)"],
        "throughput_reviews_per_sec": cost["Throughput (reviews/s)"],
    }
}
with open("results/metrics/xgboost_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

# ── Figures ───────────────────────────────────────────────────
sns.set_style("whitegrid")

# F1 comparison: in-dist vs cross-gen
fig, ax = plt.subplots(figsize=(7, 4))
metrics = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
vals_id = [res_id["accuracy"], res_id["precision"], res_id["recall"],
           res_id["f1"], res_id["auc"]]
vals_cg = [res_cg["accuracy"], res_cg["precision"], res_cg["recall"],
           res_cg["f1"], res_cg["auc"]]
x = np.arange(len(metrics))
w = 0.35
bars1 = ax.bar(x - w/2, vals_id, w, label="In-Distribution",   color="steelblue",  edgecolor="black")
bars2 = ax.bar(x + w/2, vals_cg, w, label="Cross-Generator",   color="tomato",     edgecolor="black")
ax.bar_label(bars1, fmt="%.2f", padding=2, fontsize=8)
ax.bar_label(bars2, fmt="%.2f", padding=2, fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(metrics)
ax.set_ylim(0, 1.15)
ax.set_ylabel("Score")
ax.set_title("XGBoost: In-Distribution vs Cross-Generator Performance")
ax.legend()
plt.tight_layout()
plt.savefig("results/figures/xgboost_indist_vs_crossgen.png", dpi=150)
plt.close()

# Label distribution
fig, ax = plt.subplots(figsize=(6, 4))
gen_counts = df["generator"].value_counts()
gen_counts.plot(kind="bar", ax=ax, color=sns.color_palette("Set2", len(gen_counts)),
                edgecolor="black")
ax.set_title("Review Count by Generator")
ax.set_xlabel("Generator"); ax.set_ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("results/figures/generator_distribution.png", dpi=150)
plt.close()

# Review length distribution
df["word_count"] = df["text"].str.split().str.len()
fig, ax = plt.subplots(figsize=(7, 4))
for gen, grp in df.groupby("generator"):
    ax.hist(grp["word_count"], bins=25, alpha=0.5, label=gen)
ax.set_xlabel("Word Count"); ax.set_ylabel("Frequency")
ax.set_title("Review Length Distribution by Generator")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("results/figures/length_by_generator.png", dpi=150)
plt.close()

# Cross-gen drop chart
fig, ax = plt.subplots(figsize=(5, 4))
ax.bar(["In-Distribution\nF1", f"Cross-Generator\nF1 (unseen {held_out})"],
       [res_id["f1"], res_cg["f1"]],
       color=["steelblue", "tomato"], edgecolor="black", width=0.5)
ax.axhline(y=res_id["f1"], color="steelblue", linestyle="--", alpha=0.4)
for i, v in enumerate([res_id["f1"], res_cg["f1"]]):
    ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontweight="bold")
ax.set_ylim(0, 1.15)
ax.set_ylabel("F1 Score")
ax.set_title(f"Cross-Generator Performance Drop: {drop:.3f}")
plt.tight_layout()
plt.savefig("results/figures/cross_gen_drop.png", dpi=150)
plt.close()

print("\n" + "="*60)
print("  PIPELINE COMPLETE — RESULTS")
print("="*60)
print(f"\n  Dataset       : {len(df):,} reviews ({len(df['generator'].unique())} generators)")
print(f"  In-dist  Acc  : {res_id['accuracy']:.4f}")
print(f"  In-dist  F1   : {res_id['f1']:.4f}")
print(f"  In-dist  AUC  : {res_id['auc']:.4f}")
print(f"  Cross-gen F1  : {res_cg['f1']:.4f}")
print(f"  Cross-gen drop: {drop:.4f}")
print(f"  Latency       : {cost['Latency/review (ms)']} ms/review")
print(f"\n  Results → results/")
print("="*60)
