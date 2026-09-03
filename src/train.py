"""
src/train.py
Stage 3-4: trains the FusionClassifierHead on cached CLIP embeddings.

Since CLIP is frozen and embeddings are precomputed (outputs/embeddings/*.npz),
this trains fast -- no vision/text backbone forward passes needed here at all.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

from src.model import FusionClassifierHead
from src.features import get_device

# Resolve paths relative to the project root (parent of src/), not the
# caller's working directory -- so this works whether invoked via
# `python -m src.train` from the root, or imported from a notebook in notebooks/.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMBED_DIR = os.path.join(_PROJECT_ROOT, "outputs", "embeddings")
CHECKPOINT_DIR = os.path.join(_PROJECT_ROOT, "outputs", "checkpoints")
BATCH_SIZE = 64
EPOCHS = 25
LR = 1e-3


class CachedEmbeddingDataset(Dataset):
    """Loads a precomputed .npz split (image_embeds, text_embeds, labels)."""

    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.image_embeds = torch.tensor(data["image_embeds"], dtype=torch.float32)
        self.text_embeds = torch.tensor(data["text_embeds"], dtype=torch.float32)
        self.labels = torch.tensor(data["labels"], dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.image_embeds[idx], self.text_embeds[idx], self.labels[idx]


def compute_pos_weight(labels):
    """pos_weight for BCEWithLogitsLoss = (#negatives / #positives), matches EDA's ~1.82."""
    n_pos = labels.sum().item()
    n_neg = len(labels) - n_pos
    return torch.tensor(n_neg / n_pos, dtype=torch.float32)


def evaluate(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for image_embeds, text_embeds, labels in loader:
            image_embeds, text_embeds = image_embeds.to(device), text_embeds.to(device)
            logits = model(image_embeds, text_embeds).cpu()
            all_logits.append(logits)
            all_labels.append(labels)

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    probs = torch.sigmoid(logits).numpy()
    preds = (probs >= 0.5).astype(int)
    labels_np = labels.numpy().astype(int)

    return {
        "auroc": roc_auc_score(labels_np, probs),
        "accuracy": accuracy_score(labels_np, preds),
        "macro_f1": f1_score(labels_np, preds, average="macro"),
    }


def train():
    device = get_device()
    print(f"Training on device: {device}")

    train_ds = CachedEmbeddingDataset(os.path.join(EMBED_DIR, "train_embeddings.npz"))
    val_ds = CachedEmbeddingDataset(os.path.join(EMBED_DIR, "validation_embeddings.npz"))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = FusionClassifierHead().to(device)
    pos_weight = compute_pos_weight(train_ds.labels).to(device)
    print(f"pos_weight (for class imbalance): {pos_weight.item():.3f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_val_auroc = 0.0
    history = {"train_loss": [], "val_auroc": [], "val_accuracy": [], "val_macro_f1": []}

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for image_embeds, text_embeds, labels in train_loader:
            image_embeds = image_embeds.to(device)
            text_embeds = text_embeds.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(image_embeds, text_embeds)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(labels)

        epoch_loss /= len(train_ds)
        val_metrics = evaluate(model, val_loader, device)

        history["train_loss"].append(epoch_loss)
        history["val_auroc"].append(val_metrics["auroc"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_macro_f1"].append(val_metrics["macro_f1"])

        print(f"Epoch {epoch:2d}/{EPOCHS} | train_loss={epoch_loss:.4f} | "
              f"val_auroc={val_metrics['auroc']:.4f} | val_acc={val_metrics['accuracy']:.4f} | "
              f"val_macro_f1={val_metrics['macro_f1']:.4f}")

        if val_metrics["auroc"] > best_val_auroc:
            best_val_auroc = val_metrics["auroc"]
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_model.pt"))
            print(f"  -> New best val AUROC ({best_val_auroc:.4f}), checkpoint saved.")

    print(f"\nTraining complete. Best val AUROC: {best_val_auroc:.4f}")
    print(f"Best checkpoint saved to {os.path.join(CHECKPOINT_DIR, 'best_model.pt')}")
    return model, history


if __name__ == "__main__":
    train()
