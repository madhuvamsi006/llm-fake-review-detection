"""
Leakage-controlled, product- and generator-aware train/val/test splitter.

Key rules:
  - Same product category does NOT appear in more than one split.
  - Cross-generator split: training set excludes a held-out generator;
    that generator appears only in the cross-gen test set.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional


def product_aware_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1."

    categories = df["category"].unique()
    rng = np.random.default_rng(random_seed)
    rng.shuffle(categories)

    n = len(categories)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_cats = set(categories[:n_train])
    val_cats = set(categories[n_train: n_train + n_val])
    test_cats = set(categories[n_train + n_val:])

    train = df[df["category"].isin(train_cats)].copy()
    val = df[df["category"].isin(val_cats)].copy()
    test = df[df["category"].isin(test_cats)].copy()

    _verify_no_leakage(train, val, test)
    _print_split_stats(train, val, test)
    return train, val, test


def cross_generator_split(
    df: pd.DataFrame,
    held_out_generator: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns (train, val, in_dist_test, cross_gen_test).
    held_out_generator reviews are removed from train/val/in_dist_test
    and reserved as the cross-generator test set.
    """
    cross_gen_test = df[df["generator"] == held_out_generator].copy()
    remaining = df[df["generator"] != held_out_generator].copy()

    train, val, in_dist_test = product_aware_split(
        remaining, train_ratio, val_ratio, 1 - train_ratio - val_ratio, random_seed
    )

    print(f"\nCross-generator test set ({held_out_generator}): {len(cross_gen_test):,} reviews")
    return train, val, in_dist_test, cross_gen_test


def _verify_no_leakage(train, val, test):
    train_cats = set(train["category"].unique())
    val_cats = set(val["category"].unique())
    test_cats = set(test["category"].unique())

    assert len(train_cats & val_cats) == 0, "Leakage: train/val share categories!"
    assert len(train_cats & test_cats) == 0, "Leakage: train/test share categories!"
    assert len(val_cats & test_cats) == 0, "Leakage: val/test share categories!"
    print("Split integrity check PASSED — no category overlap across splits.")


def _print_split_stats(train, val, test):
    for name, split in [("Train", train), ("Val", val), ("Test", test)]:
        pos = split["label"].sum()
        print(f"{name}: {len(split):,} reviews | genuine={len(split)-pos} | LLM={pos}")
