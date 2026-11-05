"""Offline transformer-teacher experiments for YorubaLiteSent.

Stages:
  dapt      Continue masked-language-model pretraining on MENYO Yoruba text.
  finetune  Fine-tune an African-language transformer on official YOSM splits.

Examples:
  python research/yoruba_transformer.py dapt --output_dir research_outputs/dapt
  python research/yoruba_transformer.py finetune --checkpoint research_outputs/dapt --seed 42

The transformer is an offline teacher, never the deployed lightweight model.
After fine-tuning, the script exports teacher probabilities for the YOSM training
split. The student uses those probabilities only while training.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocessing import clean_text, load_yosm_splits


DEFAULT_MODEL = "Davlan/afro-xlmr-base"
MAX_LENGTH = 256


def require_transformer_stack():
    """Import optional research dependencies only when an experiment is launched."""
    try:
        from datasets import Dataset
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import (
            AutoModelForMaskedLM,
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
            DataCollatorWithPadding,
            EarlyStoppingCallback,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise SystemExit(
            "Install the research environment first: pip install -r requirements-research.txt"
        ) from exc
    return {
        "Dataset": Dataset,
        "LoraConfig": LoraConfig,
        "TaskType": TaskType,
        "get_peft_model": get_peft_model,
        "AutoModelForMaskedLM": AutoModelForMaskedLM,
        "AutoModelForSequenceClassification": AutoModelForSequenceClassification,
        "AutoTokenizer": AutoTokenizer,
        "DataCollatorForLanguageModeling": DataCollatorForLanguageModeling,
        "DataCollatorWithPadding": DataCollatorWithPadding,
        "EarlyStoppingCallback": EarlyStoppingCallback,
        "Trainer": Trainer,
        "TrainingArguments": TrainingArguments,
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_menyo_yoruba() -> list[str]:
    """Load only MENYO training Yoruba sentences for unsupervised adaptation."""
    path = ROOT / "data" / "raw" / "menyo_train.tsv"
    if not path.exists():
        raise FileNotFoundError(f"Missing auxiliary corpus: {path}")
    texts: list[str] = []
    with path.open("r", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file, delimiter="\t"):
            text = clean_text(row.get("Yoruba", ""))
            if text:
                texts.append(text)
    if not texts:
        raise ValueError("No Yoruba MENYO training sentences were found.")
    return texts


def load_yosm_text_splits() -> dict[str, tuple[list[str], np.ndarray]]:
    """Return official YOSM splits without pooling or re-splitting them."""
    splits = load_yosm_splits()
    return {
        "train": (splits["train"][0], splits["train"][1]),
        "validation": (splits["dev"][0], splits["dev"][1]),
        "test": (splits["test"][0], splits["test"][1]),
    }


def classification_metrics(prediction):
    logits, labels = prediction
    predicted = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
    }


def make_labeled_dataset(dataset_cls, texts: list[str], labels: np.ndarray):
    return dataset_cls.from_dict({"text": texts, "labels": labels.astype(int).tolist()})


def run_dapt(args) -> None:
    """Domain-adaptive pretraining: MLM on MENYO train only, with no YOSM test text."""
    lib = require_transformer_stack()
    seed_everything(args.seed)
    texts = load_menyo_yoruba()
    tokenizer = lib["AutoTokenizer"].from_pretrained(args.base_model)
    tokenizer.model_max_length = MAX_LENGTH
    dataset = lib["Dataset"].from_dict({"text": texts})

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    model = lib["AutoModelForMaskedLM"].from_pretrained(args.base_model)
    collator = lib["DataCollatorForLanguageModeling"](tokenizer=tokenizer, mlm_probability=0.15)
    train_args = lib["TrainingArguments"](
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        save_strategy="epoch",
        logging_steps=20,
        report_to="none",
        fp16=torch.cuda.is_available(),
        seed=args.seed,
    )
    trainer = lib["Trainer"](model=model, args=train_args, train_dataset=tokenized, data_collator=collator)
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved MENYO-adapted transformer to {args.output_dir}")


def run_finetune(args) -> None:
    """Fine-tune an offline teacher and export train-set soft labels."""
    lib = require_transformer_stack()
    seed_everything(args.seed)
    splits = load_yosm_text_splits()
    tokenizer = lib["AutoTokenizer"].from_pretrained(args.checkpoint)
    tokenizer.model_max_length = MAX_LENGTH

    raw_datasets = {
        name: make_labeled_dataset(lib["Dataset"], texts, labels)
        for name, (texts, labels) in splits.items()
    }

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

    datasets = {
        name: data.map(tokenize, batched=True, remove_columns=["text"])
        for name, data in raw_datasets.items()
    }
    model = lib["AutoModelForSequenceClassification"].from_pretrained(
        args.checkpoint,
        num_labels=2,
        id2label={0: "negative", 1: "positive"},
        label2id={"negative": 0, "positive": 1},
    )
    if args.method == "lora":
        lora = lib["LoraConfig"](
            task_type=lib["TaskType"].SEQ_CLS,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["query", "value"],
        )
        model = lib["get_peft_model"](model, lora)
        model.print_trainable_parameters()

    output_dir = Path(args.output_dir) / f"{args.method}_seed_{args.seed}"
    train_args = lib["TrainingArguments"](
        output_dir=str(output_dir),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=1,
        logging_steps=10,
        report_to="none",
        fp16=torch.cuda.is_available(),
        seed=args.seed,
    )
    trainer = lib["Trainer"](
        model=model,
        args=train_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=lib["DataCollatorWithPadding"](tokenizer=tokenizer),
        compute_metrics=classification_metrics,
        callbacks=[lib["EarlyStoppingCallback"](early_stopping_patience=args.patience)],
    )
    trainer.train()

    # Teacher knowledge for distillation. These values align exactly with the
    # official YOSM training order. Validation/test labels are never exported
    # for student training.
    teacher_train = trainer.predict(datasets["train"])
    teacher_logits = teacher_train.predictions
    if isinstance(teacher_logits, tuple):
        teacher_logits = teacher_logits[0]
    shifted = teacher_logits - np.max(teacher_logits, axis=-1, keepdims=True)
    teacher_probs = np.exp(shifted) / np.exp(shifted).sum(axis=-1, keepdims=True)
    prediction = trainer.predict(datasets["test"])
    test_metrics = classification_metrics((prediction.predictions, prediction.label_ids))
    report = {
        "base_or_adapted_checkpoint": args.checkpoint,
        "method": args.method,
        "seed": args.seed,
        "max_length": MAX_LENGTH,
        "epochs_requested": args.epochs,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "test": test_metrics,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    np.savez(
        output_dir / "teacher_train_probabilities.npz",
        positive_probability=teacher_probs[:, 1].astype(np.float32),
        labels=splits["train"][1].astype(np.int64),
        seed=np.array([args.seed], dtype=np.int64),
    )
    model_dir = output_dir / "model"
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    print(json.dumps(report, indent=2))


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Yoruba transformer research experiments")
    sub = parser.add_subparsers(dest="stage", required=True)

    dapt = sub.add_parser("dapt", help="adapt an African language model to MENYO Yoruba text")
    dapt.add_argument("--base_model", default=DEFAULT_MODEL)
    dapt.add_argument("--output_dir", default="research_outputs/dapt_afroxlmr")
    dapt.add_argument("--epochs", type=float, default=5)
    dapt.add_argument("--batch_size", type=int, default=4)
    dapt.add_argument("--gradient_accumulation", type=int, default=4)
    dapt.add_argument("--learning_rate", type=float, default=5e-5)
    dapt.add_argument("--seed", type=int, default=42)
    dapt.set_defaults(func=run_dapt)

    tune = sub.add_parser("finetune", help="fine-tune and evaluate on official YOSM")
    tune.add_argument("--checkpoint", default=DEFAULT_MODEL)
    tune.add_argument("--output_dir", default="research_outputs/yosm_transformer")
    tune.add_argument("--method", choices=["lora", "full"], default="lora")
    tune.add_argument("--epochs", type=float, default=12)
    tune.add_argument("--batch_size", type=int, default=4)
    tune.add_argument("--gradient_accumulation", type=int, default=4)
    tune.add_argument("--learning_rate", type=float, default=2e-5)
    tune.add_argument("--patience", type=int, default=3)
    tune.add_argument("--lora_rank", type=int, default=8)
    tune.add_argument("--lora_alpha", type=int, default=16)
    tune.add_argument("--lora_dropout", type=float, default=0.1)
    tune.add_argument("--seed", type=int, default=42)
    tune.set_defaults(func=run_finetune)
    return parser


if __name__ == "__main__":
    args = make_parser().parse_args()
    args.func(args)
