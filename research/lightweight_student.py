"""YorubaLiteSent: lightweight FastText-initialized, distilled sentiment student.

The deployable student is a compact CNN-BiLSTM-Attention model. A transformer
is optional and is used only to provide training-set soft labels. The student
uses a pure YOSM vocabulary, so MENYO text transfers through subword embeddings,
not through a large randomly initialized vocabulary.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocessing import MAX_LEN, MIN_FREQ, build_vocab, clean_text, load_yosm_splits, tokenize, encode_texts
from cnn_bilstm_attention import CNNBiLSTMAttention


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBED_DIM = 200


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_menyo_sentences() -> list[list[str]]:
    """Read Yoruba-side MENYO training text for unsupervised subword embedding learning."""
    path = ROOT / "data" / "raw" / "menyo_train.tsv"
    if not path.exists():
        raise FileNotFoundError(f"Missing MENYO file: {path}")
    sentences: list[list[str]] = []
    with path.open("r", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file, delimiter="\t"):
            text = clean_text(row.get("Yoruba", ""))
            tokens = tokenize(text)
            if tokens:
                sentences.append(tokens)
    if not sentences:
        raise ValueError("No Yoruba sentences were read from MENYO training data.")
    return sentences


def train_or_load_fasttext(path: Path, seed: int, epochs: int):
    """Train compact FastText embeddings; FastText builds vectors from subwords."""
    try:
        from gensim.models import FastText
    except ImportError as exc:
        raise SystemExit("Install research dependencies: pip install -r requirements-research.txt") from exc
    if path.exists():
        return FastText.load(str(path))
    sentences = load_menyo_sentences()
    model = FastText(
        sentences=sentences,
        vector_size=EMBED_DIM,
        window=5,
        min_count=1,
        workers=1,
        sg=1,
        seed=seed,
        epochs=epochs,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))
    return model


def make_embedding_matrix(vocab: dict[str, int], fasttext_model, seed: int) -> np.ndarray:
    """Initialize the small YOSM vocabulary using MENYO FastText subword vectors."""
    rng = np.random.default_rng(seed)
    matrix = rng.normal(0.0, 0.02, size=(len(vocab), EMBED_DIM)).astype(np.float32)
    matrix[vocab["<pad>"]] = 0.0
    for token, index in vocab.items():
        if token in {"<pad>", "<unk>"}:
            continue
        try:
            matrix[index] = fasttext_model.wv.get_vector(token)
        except KeyError:
            # The random initialization is retained only if vector creation fails.
            pass
    return matrix


def metrics(labels: np.ndarray, predicted: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, predicted, labels=[0, 1]).tolist(),
    }


@torch.no_grad()
def evaluate(model, loader, loss_fn: nn.Module | None = None) -> dict:
    model.eval()
    all_labels, all_predicted, total_loss, count = [], [], 0.0, 0
    for batch in loader:
        x, y = batch[:2]
        logits = model(x.to(DEVICE))
        if loss_fn is not None:
            total_loss += loss_fn(logits, y.to(DEVICE)).item() * len(x)
            count += len(x)
        predicted = (torch.sigmoid(logits) >= 0.5).long().cpu().numpy()
        all_labels.extend(y.numpy().astype(int))
        all_predicted.extend(predicted)
    result = metrics(np.asarray(all_labels), np.asarray(all_predicted))
    if loss_fn is not None:
        result["loss"] = total_loss / max(count, 1)
    return result


def benchmark_cpu(model, sample: torch.Tensor) -> dict:
    """Measure CPU latency for the final student and its dynamic-quantized form."""
    def timed(candidate, loops=100):
        candidate.eval()
        with torch.no_grad():
            for _ in range(10):
                candidate(sample)
            start = time.perf_counter()
            for _ in range(loops):
                candidate(sample)
        return round((time.perf_counter() - start) * 1000 / loops, 3)

    cpu_model = copy.deepcopy(model).cpu()
    results = {"fp32_batch1_latency_ms": timed(cpu_model)}
    try:
        quantized = torch.quantization.quantize_dynamic(cpu_model, {nn.LSTM, nn.Linear}, dtype=torch.qint8)
        results["dynamic_int8_batch1_latency_ms"] = timed(quantized)
    except (RuntimeError, AttributeError) as error:
        results["quantization_note"] = str(error)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a lightweight distilled Yoruba sentiment student")
    parser.add_argument("--output_dir", default="research_outputs/yorubalitesent")
    parser.add_argument("--fasttext_model_path", default=None,
                        help="Reuse an existing MENYO FastText model rather than training another copy.")
    parser.add_argument("--teacher_probs", default=None,
                        help="Path to teacher_train_probabilities.npz. Omit for FastText-only ablation.")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=3e-3)
    parser.add_argument("--filters", type=int, default=32,
                        help="CNN filters per kernel; 32 keeps the student compact.")
    parser.add_argument("--hidden_dim", type=int, default=32,
                        help="BiLSTM hidden units per direction; 32 reduces overfitting on YOSM.")
    parser.add_argument("--dropout", type=float, default=0.6)
    parser.add_argument("--embedding_dropout", type=float, default=0.15)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--freeze_embeddings_epochs", type=int, default=25,
                        help="Keep MENYO FastText embeddings fixed for these initial epochs.")
    parser.add_argument("--monitor", choices=("loss", "f1"), default="loss",
                        help="Validation metric used for the scheduler and checkpoint selection.")
    parser.add_argument("--alpha", type=float, default=0.7,
                        help="Weight for true-label BCE loss; remaining weight is distillation loss.")
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--fasttext_epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    if args.freeze_embeddings_epochs < 0:
        raise ValueError("freeze_embeddings_epochs must be non-negative")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("dropout must be in the range [0, 1)")
    if not 0.0 <= args.embedding_dropout < 1.0:
        raise ValueError("embedding_dropout must be in the range [0, 1)")
    if not 0.0 <= args.label_smoothing < 0.5:
        raise ValueError("label_smoothing must be in the range [0, 0.5)")

    seed_everything(args.seed)
    output_dir = ROOT / args.output_dir / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = load_yosm_splits()
    train_texts, y_train = splits["train"]
    val_texts, y_val = splits["dev"]
    test_texts, y_test = splits["test"]

    # Pure YOSM vocabulary: no test tokens and no unused MENYO word rows.
    vocab, own_vocab_size, _ = build_vocab([tokenize(text) for text in train_texts], MIN_FREQ)
    with (output_dir / "vocab.pkl").open("wb") as file:
        pickle.dump(vocab, file)
    x_train = encode_texts(train_texts, vocab, MAX_LEN)
    x_val = encode_texts(val_texts, vocab, MAX_LEN)
    x_test = encode_texts(test_texts, vocab, MAX_LEN)

    fasttext_path = Path(args.fasttext_model_path) if args.fasttext_model_path else ROOT / args.output_dir / "menyo_yoruba_fasttext.model"
    fasttext = train_or_load_fasttext(fasttext_path, args.seed, args.fasttext_epochs)
    embedding_matrix = make_embedding_matrix(vocab, fasttext, args.seed)

    teacher = None
    if args.teacher_probs:
        teacher_data = np.load(args.teacher_probs)
        teacher = teacher_data["positive_probability"].astype(np.float32)
        if len(teacher) != len(y_train) or not np.array_equal(teacher_data["labels"], y_train):
            raise ValueError("Teacher probabilities do not align with the official YOSM training split.")

    train_tensors = [torch.from_numpy(x_train).long(), torch.from_numpy(y_train).float()]
    if teacher is not None:
        train_tensors.append(torch.from_numpy(teacher))
    train_loader = DataLoader(TensorDataset(*train_tensors), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(x_val).long(), torch.from_numpy(y_val).float()),
                            batch_size=args.batch_size)
    test_loader = DataLoader(TensorDataset(torch.from_numpy(x_test).long(), torch.from_numpy(y_test).float()),
                             batch_size=args.batch_size)

    model = CNNBiLSTMAttention(
        vocab_size=len(vocab),
        filters=args.filters,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        classifier_dropout=args.dropout,
        embedding_dropout=args.embedding_dropout,
    ).to(DEVICE)
    with torch.no_grad():
        model.embedding.weight.copy_(torch.from_numpy(embedding_matrix).to(DEVICE))
    # Do not let the small labelled YOSM set immediately overwrite MENYO knowledge.
    model.embedding.weight.requires_grad = args.freeze_embeddings_epochs == 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min" if args.monitor == "loss" else "max", factor=0.5, patience=2
    )
    hard_loss_fn = nn.BCEWithLogitsLoss()
    best_state, best_epoch = None, 0
    best_score = float("inf") if args.monitor == "loss" else -1.0
    best_f1, waiting = -1.0, 0
    history = {"train_loss": [], "validation_loss": [], "validation_f1": [], "learning_rate": []}

    for epoch in range(1, args.epochs + 1):
        if epoch == args.freeze_embeddings_epochs + 1 and not model.embedding.weight.requires_grad:
            model.embedding.weight.requires_grad = True
            print("Unfroze MENYO FastText embeddings for fine-tuning.")
        model.train()
        total_loss, count = 0.0, 0
        for batch in train_loader:
            x, labels = batch[0].to(DEVICE), batch[1].to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            smoothed_labels = labels * (1 - 2 * args.label_smoothing) + args.label_smoothing
            hard_loss = hard_loss_fn(logits, smoothed_labels)
            if teacher is None:
                loss = hard_loss
            else:
                teacher_probability = batch[2].to(DEVICE).clamp(1e-6, 1 - 1e-6)
                teacher_logit = torch.logit(teacher_probability)
                softened_teacher = torch.sigmoid(teacher_logit / args.temperature)
                soft_loss = functional.binary_cross_entropy_with_logits(
                    logits / args.temperature, softened_teacher
                ) * (args.temperature ** 2)
                loss = args.alpha * hard_loss + (1 - args.alpha) * soft_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item() * len(x)
            count += len(x)

        validation = evaluate(model, val_loader, hard_loss_fn)
        monitored_value = validation["loss"] if args.monitor == "loss" else validation["f1"]
        scheduler.step(monitored_value)
        history["train_loss"].append(total_loss / max(count, 1))
        history["validation_loss"].append(validation["loss"])
        history["validation_f1"].append(validation["f1"])
        history["learning_rate"].append(optimizer.param_groups[0]["lr"])
        improved = monitored_value < best_score if args.monitor == "loss" else monitored_value > best_score
        if improved:
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            best_epoch, best_score, best_f1, waiting = epoch, monitored_value, validation["f1"], 0
        else:
            waiting += 1
        print(
            f"epoch {epoch:02d} | train_loss {history['train_loss'][-1]:.4f} "
            f"| val_loss {validation['loss']:.4f} | val_f1 {validation['f1']:.4f} "
            f"| lr {history['learning_rate'][-1]:.1e}"
        )
        if waiting >= args.patience:
            break

    model.load_state_dict(best_state)
    test = evaluate(model, test_loader)
    state_path = output_dir / "lightweight_student.pt"
    torch.save({"state_dict": model.state_dict(), "vocab_size": len(vocab)}, state_path)
    param_count = sum(parameter.numel() for parameter in model.parameters())
    report = {
        "model": "YorubaLiteSent-CNNBiLSTMAttention",
        "teacher_distillation": teacher is not None,
        "vocabulary": "pure_yosm_train_only",
        "vocab_size": len(vocab),
        "embedding_source": "MENYO Yoruba FastText subword embeddings",
        "regularization": {
            "filters_per_kernel": args.filters,
            "bilstm_hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "embedding_dropout": args.embedding_dropout,
            "label_smoothing": args.label_smoothing,
            "weight_decay": args.weight_decay,
            "initial_learning_rate": args.lr,
            "scheduler": f"ReduceLROnPlateau(validation_{args.monitor}, factor=0.5, patience=2)",
            "freeze_embeddings_epochs": args.freeze_embeddings_epochs,
            "early_stopping_patience": args.patience,
            "checkpoint_monitor": f"validation_{args.monitor}",
        },
        "params": param_count,
        "checkpoint_size_mb": round(state_path.stat().st_size / (1024 * 1024), 3),
        "best_epoch": best_epoch,
        "validation_f1": best_f1,
        "test": test,
        "history": history,
        "seed": args.seed,
        "device": str(DEVICE),
    }
    if args.benchmark:
        report["cpu_benchmark"] = benchmark_cpu(model, torch.from_numpy(x_test[:1]).long())
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
