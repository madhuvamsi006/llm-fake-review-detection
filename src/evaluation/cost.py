"""
Inference cost benchmarking: latency, throughput, and peak memory usage.
All three classifiers are benchmarked on the same fixed hardware configuration.
"""
import time
import psutil
import os
import numpy as np
import pandas as pd
from pathlib import Path


def benchmark_xgboost(model, X, n_runs: int = 3) -> dict:
    latencies = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = model.predict_proba(X)
        latencies.append(time.perf_counter() - start)

    n_reviews = X.shape[0]
    avg_latency = np.mean(latencies)
    return _build_report("XGBoost", avg_latency, n_reviews)


def benchmark_transformer(model, dataset, batch_size: int = 32, n_runs: int = 3, label: str = "BERT") -> dict:
    import torch
    from torch.utils.data import DataLoader
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    loader = DataLoader(dataset, batch_size=batch_size)
    process = psutil.Process(os.getpid())

    latencies = []
    peak_mem_mb = 0

    for _ in range(n_runs):
        start = time.perf_counter()
        with torch.no_grad():
            for batch in loader:
                _ = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                )
        latencies.append(time.perf_counter() - start)
        mem = process.memory_info().rss / (1024 ** 2)
        peak_mem_mb = max(peak_mem_mb, mem)

    n_reviews = len(dataset)
    avg_latency = np.mean(latencies)
    return _build_report(label, avg_latency, n_reviews, peak_mem_mb)


def _build_report(label: str, total_latency_s: float, n_reviews: int, peak_mem_mb: float = None) -> dict:
    latency_per_review_ms = (total_latency_s / n_reviews) * 1000
    throughput = n_reviews / total_latency_s
    report = {
        "Classifier": label,
        "Total latency (s)": round(total_latency_s, 3),
        "Latency/review (ms)": round(latency_per_review_ms, 4),
        "Throughput (reviews/s)": round(throughput, 1),
        "Peak RAM (MB)": round(peak_mem_mb, 1) if peak_mem_mb else "N/A",
        "n_reviews": n_reviews,
    }
    print(f"\n[Cost] {label}: {latency_per_review_ms:.4f} ms/review | "
          f"{throughput:.1f} reviews/s | RAM: {peak_mem_mb or 'N/A'} MB")
    return report


def cost_table(reports: list) -> pd.DataFrame:
    df = pd.DataFrame(reports)
    save_path = Path("results/cost_analysis/cost_comparison.csv")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"\nCost table saved to {save_path}")
    return df
