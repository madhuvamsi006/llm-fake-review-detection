"""
Tokenizer wrappers and HuggingFace Dataset builders for BERT and DeBERTa.
"""
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class ReviewDataset(Dataset):
    def __init__(self, texts: list, labels: list, tokenizer, max_length: int = 512):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


def build_dataset(
    texts: list,
    labels: list,
    model_name: str,
    max_length: int = 512,
) -> ReviewDataset:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return ReviewDataset(texts, labels, tokenizer, max_length)


def get_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name)
