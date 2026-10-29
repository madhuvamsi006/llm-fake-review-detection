"""
Fine-tuned BERT classifier for fake review detection.
"""
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from tqdm import tqdm

MODEL_NAME = "bert-base-uncased"


def train_bert(
    train_dataset,
    val_dataset,
    model_name: str = MODEL_NAME,
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 2e-5,
    save_path: str = "results/checkpoints/bert/",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
    )

    best_val_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_loss / len(train_loader)

        val_loss = _evaluate_loss(model, val_loader, device)
        print(f"Epoch {epoch+1} | train_loss={avg_train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            Path(save_path).mkdir(parents=True, exist_ok=True)
            model.save_pretrained(save_path)
            AutoTokenizer.from_pretrained(model_name).save_pretrained(save_path)
            print(f"  Saved best model to {save_path}")

    return model


def _evaluate_loss(model, loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            total_loss += outputs.loss.item()
    return total_loss / len(loader)


def predict(model, dataset, batch_size: int = 32) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size)
    all_probs = []
    with torch.no_grad():
        for batch in loader:
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            probs = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
    return np.array(all_probs)


def load_model(path: str = "results/checkpoints/bert/"):
    model = AutoModelForSequenceClassification.from_pretrained(path)
    return model
