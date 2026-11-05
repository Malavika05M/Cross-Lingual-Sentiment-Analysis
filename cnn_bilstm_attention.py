"""
STEP 3 - CNN-BiLSTM-ATTENTION HYBRID MODEL PIPELINE
Comparative Model for Sentiment Classification on SST-2 and YOSM

Architecture:
    1. Trainable Embedding Layer (vocab_size -> emb_dim)
    2. Parallel 1D Multi-Kernel Convolutions (kernels 2, 3, 4, 5, padding='same')
       to capture local n-gram patterns across temporal positions.
    3. Bidirectional LSTM (BiLSTM) capturing long-range contextual dependencies.
    4. Masked Self-Attention Pooling mechanism that scores tokens while ignoring padding.
    5. Regularized Classification Head with Dropout and Dense layers -> binary logit.

Outputs:
    - Checkpoint: results/{dataset}/cnn_bilstm_attention.pt
    - Metrics JSON: results/{dataset}/cnn_bilstm_metrics.json
    - Training curves: results/{dataset}/bilstm_training_curves.png
    - Confusion matrix: results/{dataset}/bilstm_confusion_matrix.png
"""

import os
import json
import time
import pickle
import random
import argparse
import warnings
warnings.filterwarnings("ignore", message=".*Using padding='same'.*")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

SEED         = 42
DATA_ROOT    = "data/processed"
RESULTS_ROOT = "results"
DATASETS     = ("yosm", "sst2")

EMB_DIM      = 200
FILTERS      = 100            # filters per conv kernel
KERNELS      = (2, 3, 4, 5)   # local n-gram feature extractors
HIDDEN_DIM   = 128            # BiLSTM hidden size per direction (256 bidirectional)
DROPOUT      = 0.5
LR           = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE   = 64
EPOCHS       = 20
PATIENCE     = 5
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int = SEED):
    """Seed all random number generators for strict reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MaskedAttention(nn.Module):
    """Additive attention mechanism with padding mask support."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1, bias=False)
        )

    def forward(self, h, mask=None):
        # h: (B, L, hidden_dim)
        scores = self.attn(h).squeeze(-1)  # (B, L)
        if mask is not None:
            # Mask out padding tokens with large negative value
            scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=-1)  # (B, L)
        # Weighted context sum: (B, 1, L) x (B, L, hidden_dim) -> (B, hidden_dim)
        context = torch.bmm(weights.unsqueeze(1), h).squeeze(1)
        return context, weights


class CNNBiLSTMAttention(nn.Module):
    """Deep Hybrid: Trainable Embedding -> Parallel CNNs -> BiLSTM -> Masked Attention -> Head."""
    def __init__(self, vocab_size, emb_dim=EMB_DIM, kernels=KERNELS,
                 filters=FILTERS, hidden_dim=HIDDEN_DIM, dropout=DROPOUT,
                 classifier_dropout=0.3, embedding_dropout=0.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        self.convs = nn.ModuleList([
            nn.Conv1d(emb_dim, filters, kernel_size=k, padding="same")
            for k in kernels
        ])
        conv_out_dim = filters * len(kernels)
        self.lstm = nn.LSTM(
            input_size=conv_out_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.attention = MaskedAttention(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: (B, L) token IDs
        mask = (x != 0)                        # True for non-pad tokens
        e = self.embedding_dropout(self.embedding(x)).transpose(1, 2)  # (B, emb_dim, L)
        # Parallel convolutions with padding='same' preserve sequence length L
        c_feats = [torch.relu(conv(e)) for conv in self.convs]
        c_cat = torch.cat(c_feats, dim=1).transpose(1, 2)  # (B, L, conv_out_dim)
        # Bidirectional sequential modeling
        lstm_out, _ = self.lstm(c_cat)                     # (B, L, 2 * hidden_dim)
        # Sequence-level attention pooling respecting padding
        context, attn_weights = self.attention(lstm_out, mask)  # (B, 2 * hidden_dim)
        h = self.dropout(context)
        logits = self.classifier(h).squeeze(-1)            # (B,)
        return logits


def load_split(name, data_dir):
    """Load preprocessed split (.npz) containing x and y."""
    d = np.load(os.path.join(data_dir, f"{name}.npz"))
    return d["x"], d["y"].astype(np.float32)


def make_loader(x, y, batch_size=BATCH_SIZE, shuffle=False):
    """Construct PyTorch DataLoader from numpy arrays."""
    tx = torch.from_numpy(x).long()
    ty = torch.from_numpy(y)
    return DataLoader(TensorDataset(tx, ty), batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def run_eval(model, loader, loss_fn=None):
    """Evaluate model on DataLoader and return classification metrics and loss."""
    model.eval()
    ys, preds, losses, n = [], [], [], 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        logits = model(xb)
        if loss_fn is not None:
            losses.append(loss_fn(logits, yb).item() * len(xb))
            n += len(xb)
        probs = torch.sigmoid(logits).cpu().numpy()
        ys.extend(yb.cpu().numpy().astype(int))
        preds.extend((probs >= 0.5).astype(int))

    y_true, y_pred = np.asarray(ys), np.asarray(preds)
    out = {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }
    if loss_fn is not None:
        out["loss"] = float(sum(losses) / max(n, 1))
    return out, y_true, y_pred


def train_model(model, train_loader, val_loader, y_train, epochs=EPOCHS,
                patience=PATIENCE, lr=LR, weight_decay=WEIGHT_DECAY,
                monitor="f1"):
    """Train hybrid model with AdamW, BCEWithLogitsLoss, and early stopping."""
    neg, pos = float((y_train == 0).sum()), float((y_train == 1).sum())
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}
    best_score = 0.0 if monitor == "f1" else float("inf")
    best_state = None
    best_epoch = 0
    wait = 0

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running += loss.item() * len(xb)
            seen += len(xb)

        val, _, _ = run_eval(model, val_loader, loss_fn)
        train_loss = running / max(seen, 1)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val["loss"])
        history["val_acc"].append(val["accuracy"])
        history["val_f1"].append(val["f1"])

        current_score = val["f1"] if monitor == "f1" else val["loss"]
        improved = (current_score > best_score) if monitor == "f1" else (current_score < best_score)

        marker = ""
        if improved:
            best_score = current_score
            best_epoch = epoch
            wait = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            marker = "  <- best"
        else:
            wait += 1

        print(f"epoch {epoch:02d} | train_loss {train_loss:.4f} "
              f"| val_loss {val['loss']:.4f} | val_acc {val['accuracy']:.4f} "
              f"| val_f1 {val['f1']:.4f}{marker}")

        if wait >= patience:
            print(f"Early stopping triggered (no improvement for {patience} epochs).")
            break

    total_time = time.time() - t0
    return history, total_time, best_state, best_epoch


def plot_history(history, results_dir, prefix="bilstm"):
    """Plot and save training/validation loss and metric curves."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].plot(history["train_loss"], label="train", color="#1f77b4")
    axes[0].plot(history["val_loss"], label="val", color="#ff7f0e")
    axes[0].set_title("Hybrid Model Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].plot(history["val_acc"], label="val acc", color="#2ca02c")
    axes[1].plot(history["val_f1"], label="val F1", color="#d62728")
    axes[1].set_title("Hybrid Validation Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, f"{prefix}_training_curves.png"), dpi=150)
    plt.close(fig)


def plot_cm(cm, results_dir, prefix="bilstm"):
    """Plot and save confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Greens", interpolation="nearest")
    thresh = cm.max() / 2 if cm.max() > 0 else 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Negative (0)", "Positive (1)"])
    ax.set_yticklabels(["Negative (0)", "Positive (1)"])
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"Confusion Matrix ({prefix.upper()})")
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, f"{prefix}_confusion_matrix.png"), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="CNN-BiLSTM-Attention Sentiment Model")
    parser.add_argument("--dataset", choices=DATASETS, default="yosm",
                        help="Preprocessed dataset to train/evaluate on (yosm or sst2)")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Maximum training epochs")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=LR, help="Learning rate")
    parser.add_argument("--hidden_dim", type=int, default=HIDDEN_DIM, help="BiLSTM hidden dimension")
    parser.add_argument("--patience", type=int, default=PATIENCE, help="Early stopping patience")
    parser.add_argument("--monitor", choices=["f1", "loss"], default="f1",
                        help="Metric to monitor for early stopping")
    parser.add_argument("--eval_only", action="store_true",
                        help="Only evaluate existing checkpoint without retraining")
    args = parser.parse_args()

    data_dir    = os.path.join(DATA_ROOT, args.dataset)
    results_dir = os.path.join(RESULTS_ROOT, args.dataset)

    seed_everything(SEED)
    os.makedirs(results_dir, exist_ok=True)

    x_tr, y_tr = load_split("train", data_dir)
    x_va, y_va = load_split("val", data_dir)
    x_te, y_te = load_split("test", data_dir)
    print(f"[{args.dataset.upper()}] train: {len(x_tr)} | val: {len(x_va)} | test: {len(x_te)} | device: {DEVICE}")

    vocab_file = os.path.join(data_dir, "vocab.pkl")
    with open(vocab_file, "rb") as f:
        vocab = pickle.load(f)
    vocab_size = max(vocab.values()) + 1

    train_loader = make_loader(x_tr, y_tr, batch_size=args.batch_size, shuffle=True)
    val_loader   = make_loader(x_va, y_va, batch_size=args.batch_size, shuffle=False)
    test_loader  = make_loader(x_te, y_te, batch_size=args.batch_size, shuffle=False)

    model = CNNBiLSTMAttention(vocab_size, hidden_dim=args.hidden_dim).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"CNN-BiLSTM-Attention | vocab: {vocab_size} | trainable parameters: {n_params:,}\n")

    ckpt_path = os.path.join(results_dir, "cnn_bilstm_attention.pt")
    metrics_path = os.path.join(results_dir, "cnn_bilstm_metrics.json")

    if args.eval_only:
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
        print(f"Loading existing checkpoint from {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt["state_dict"])
        test, y_true, y_pred = run_eval(model, test_loader)
        print("\n===== TEST RESULTS (EVAL ONLY) =====")
        print(classification_report(y_true, y_pred, target_names=["negative", "positive"], zero_division=0))
        print(f"Accuracy:  {test['accuracy']:.4f}")
        print(f"Precision: {test['precision']:.4f}")
        print(f"Recall:    {test['recall']:.4f}")
        print(f"F1 Score:  {test['f1']:.4f}")
        print(f"Confusion Matrix:\n{np.asarray(test['confusion_matrix'])}")
        return

    history, train_time, best_state, best_epoch = train_model(
        model, train_loader, val_loader, y_tr, epochs=args.epochs,
        patience=args.patience, lr=args.lr, monitor=args.monitor
    )

    if best_state is not None:
        model.load_state_dict(best_state)

    test, y_true, y_pred = run_eval(model, test_loader)

    print(f"\n===== TEST RESULTS - CNN-BiLSTM-ATTENTION ({args.dataset.upper()}) =====")
    print(classification_report(y_true, y_pred, target_names=["negative", "positive"], zero_division=0))
    print(f"Confusion Matrix:\n{np.asarray(test['confusion_matrix'])}")
    print(f"Accuracy:  {test['accuracy']:.4f}")
    print(f"Precision: {test['precision']:.4f}")
    print(f"Recall:    {test['recall']:.4f}")
    print(f"F1 Score:  {test['f1']:.4f}")
    print(f"Best Epoch: {best_epoch} | Train Time: {train_time:.1f}s")

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": "CNN-BiLSTM-Attention",
            "dataset": args.dataset,
            "params": n_params,
            "best_epoch": best_epoch,
            "train_time_sec": round(train_time, 2),
            "test": test,
            "history": history
        }, f, indent=2)

    torch.save({
        "state_dict": best_state if best_state is not None else model.state_dict(),
        "vocab_size": vocab_size,
        "config": {
            "emb_dim": EMB_DIM, "kernels": list(KERNELS),
            "filters": FILTERS, "hidden_dim": args.hidden_dim
        }
    }, ckpt_path)

    plot_history(history, results_dir, prefix="bilstm")
    plot_cm(np.asarray(test["confusion_matrix"]), results_dir, prefix="bilstm")
    print(f"\nArtifacts successfully saved to {results_dir}/")


if __name__ == "__main__":
    main()
