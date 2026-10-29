# Detecting LLM-Generated Fake Reviews on Retail Platforms

**Author:** Madhu Vamsi Turaka (1187433)  
**Supervisor:** Dr Jyoti Dabass  
**Programme:** MSc Data Science — Liverpool John Moores University  

## Overview

A leakage-controlled, cross-generator benchmark for cost-aware detection of LLM-generated fake reviews. Extends the Salminen et al. (2022) dataset with reviews from GPT-4o, Claude, and Llama-3 and compares XGBoost, BERT, and DeBERTa on in-distribution accuracy, cross-generator generalisation, and deployment cost.

## Project Structure

```
├── data/
│   ├── raw/            # Salminen dataset (genuine + GPT-2)
│   ├── generated/      # New reviews: GPT-4o, Claude, Llama-3
│   └── processed/      # Merged, split datasets
├── src/
│   ├── data/           # Loader, generator, splitter
│   ├── features/       # Classical and transformer feature pipelines
│   ├── models/         # XGBoost, BERT, DeBERTa
│   └── evaluation/     # Metrics and cost benchmarking
├── notebooks/          # Jupyter notebooks (one per experiment phase)
├── configs/            # config.yaml
└── results/            # Saved metrics, figures, cost tables
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your API keys
```

## Dataset

Base dataset: [Salminen et al. (2022) Fake Reviews Dataset](https://osf.io/tyue9/)  
New synthetic reviews generated for this study are stored in `data/generated/`.

## Research Questions

- RQ1: In-distribution detection accuracy (XGBoost vs BERT vs DeBERTa)
- RQ2: Cross-generator accuracy drop
- RQ3: Multi-generator training vs single-generator training
- RQ4: Accuracy-cost trade-off (latency + memory)
