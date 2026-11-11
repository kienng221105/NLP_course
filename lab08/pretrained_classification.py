"""Các hàm dùng chung cho bài fine-tuning BERT trong LAB 08."""

from __future__ import annotations

import copy
import html
import random
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "google/bert_uncased_L-2_H-128_A-2"
RANDOM_SEED = 42
LABEL_TO_ID = {"negative": 0, "positive": 1}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Đặt seed cho các thư viện để lần chạy có thể lặp lại gần giống nhau."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_project_dir(start: Path | None = None) -> Path:
    """Tìm thư mục dự án từ vị trí mở notebook hoặc file script."""
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "data" / "IMDB Dataset.csv").exists():
            return candidate
    raise FileNotFoundError(
        "Không tìm thấy data/IMDB Dataset.csv. Hãy mở notebook trong dự án NLP."
    )


def normalize_review(text: object) -> str:
    """Bỏ HTML và khoảng trắng thừa, giữ dấu câu để tokenizer xử lý."""
    text = html.unescape(str(text))
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_imdb(path: str | Path) -> tuple[pd.DataFrame, dict[str, int]]:
    """Đọc IMDb, bỏ dòng lỗi/trùng và trả lại thống kê các dòng đã loại."""
    raw = pd.read_csv(path, encoding="utf-8")
    required = {"review", "sentiment"}
    if not required.issubset(raw.columns):
        raise ValueError(f"IMDb cần có các cột: {sorted(required)}")

    data = raw[["review", "sentiment"]].dropna().copy()
    data["review"] = data["review"].map(normalize_review)
    data["sentiment"] = data["sentiment"].astype(str).str.strip().str.lower()
    data = data[data["review"].str.len() > 0].copy()
    unknown_labels = set(data["sentiment"].unique()) - set(LABEL_TO_ID)
    if unknown_labels:
        raise ValueError(f"Nhãn không hợp lệ: {sorted(unknown_labels)}")

    label_counts = data.groupby("review")["sentiment"].nunique()
    conflicting = set(label_counts[label_counts > 1].index)
    conflict_rows_removed = int(data["review"].isin(conflicting).sum())
    data = data[~data["review"].isin(conflicting)].copy()
    duplicate_rows_removed = int(data["review"].duplicated().sum())
    data = data.drop_duplicates(subset="review", keep="first").reset_index(drop=True)
    data["label_id"] = data["sentiment"].map(LABEL_TO_ID).astype("int64")

    stats = {
        "raw_rows": len(raw),
        "rows_after_cleaning": len(data),
        "missing_rows_removed": len(raw) - len(raw[["review", "sentiment"]].dropna()),
        "conflicting_rows_removed": conflict_rows_removed,
        "duplicate_rows_removed": duplicate_rows_removed,
    }
    return data, stats


def split_imdb(data: pd.DataFrame, seed: int = RANDOM_SEED):
    """Tách dữ liệu phân tầng thành 60% train, 20% validation, 20% test."""
    train_val, test = train_test_split(
        data, test_size=0.20, stratify=data["label_id"], random_state=seed
    )
    train, validation = train_test_split(
        train_val,
        test_size=0.25,
        stratify=train_val["label_id"],
        random_state=seed,
    )
    return (
        train.reset_index(drop=True),
        validation.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def load_tokenizer(model_name: str = MODEL_NAME):
    return AutoTokenizer.from_pretrained(model_name)


def encode_texts(tokenizer, texts, max_length: int = 256):
    """Tokenize và padding/truncate một danh sách câu thành tensor."""
    return tokenizer(
        list(texts),
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


def make_loader(encodings, labels, max_length: int, batch_size: int, shuffle: bool):
    """Tạo DataLoader từ dữ liệu đã tokenize; có thể cắt xuống độ dài ngắn hơn."""
    input_ids = encodings["input_ids"][:, :max_length].contiguous()
    attention_mask = encodings["attention_mask"][:, :max_length].contiguous()
    labels = torch.tensor(np.asarray(labels, dtype=np.int64).copy())
    if len(input_ids) != len(labels):
        raise ValueError("Số input và label không khớp.")
    dataset = TensorDataset(input_ids, attention_mask, labels)
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def build_model(
    model_name: str = MODEL_NAME,
    pretrained: bool = True,
    num_labels: int = 2,
):
    """Tạo classifier pretrained hoặc cùng kiến trúc với trọng số ngẫu nhiên."""
    if pretrained:
        return AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=ID_TO_LABEL,
            label2id=LABEL_TO_ID,
        )
    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )
    return AutoModelForSequenceClassification.from_config(config)


def freeze_encoder(model) -> None:
    """Đóng băng BERT, chỉ để classification head được cập nhật."""
    for parameter in model.base_model.parameters():
        parameter.requires_grad = False
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True


def parameter_counts(model) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return total, trainable


def evaluate_model(model, loader, device):
    """Tính loss và các metric phân loại nhị phân."""
    model.eval()
    total_loss = 0.0
    targets, predictions, probabilities = [], [], []
    with torch.no_grad():
        for input_ids, attention_mask, labels in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            probs = torch.softmax(outputs.logits, dim=-1)
            batch_size = len(labels)
            total_loss += outputs.loss.item() * batch_size
            targets.extend(labels.cpu().numpy().tolist())
            predictions.extend(probs.argmax(dim=-1).cpu().numpy().tolist())
            probabilities.extend(probs.cpu().numpy().tolist())

    metrics = {
        "loss": total_loss / max(1, len(targets)),
        "accuracy": accuracy_score(targets, predictions),
        "precision": precision_score(targets, predictions, zero_division=0),
        "recall": recall_score(targets, predictions, zero_division=0),
        "f1": f1_score(targets, predictions, zero_division=0),
        "f1_macro": f1_score(targets, predictions, average="macro", zero_division=0),
    }
    return {
        "metrics": metrics,
        "targets": np.asarray(targets, dtype=np.int64),
        "predictions": np.asarray(predictions, dtype=np.int64),
        "probabilities": np.asarray(probabilities, dtype=np.float32),
    }


def train_model(
    model,
    train_loader,
    validation_loader,
    device,
    epochs: int = 1,
    learning_rate: float = 2e-5,
):
    """Huấn luyện một số epoch và giữ checkpoint có validation loss thấp nhất."""
    from sklearn.metrics import accuracy_score, f1_score

    model.to(device)
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    if not trainable_parameters:
        raise ValueError("Không có tham số nào được phép huấn luyện.")
    optimizer = torch.optim.AdamW(trainable_parameters, lr=learning_rate)
    history = []
    best_loss = float("inf")
    best_state = None
    process = None
    try:
        import psutil
        process = psutil.Process()
        start_rss_mb = process.memory_info().rss / (1024**2)
    except ImportError:
        start_rss_mb = float("nan")
    peak_rss_mb = start_rss_mb
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        train_targets, train_predictions = [], []
        for input_ids, attention_mask, labels in train_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            outputs.loss.backward()
            optimizer.step()
            batch_size = len(labels)
            total_loss += outputs.loss.item() * batch_size
            train_targets.extend(labels.detach().cpu().numpy().tolist())
            train_predictions.extend(
                outputs.logits.detach().argmax(dim=-1).cpu().numpy().tolist()
            )
            if process is not None:
                peak_rss_mb = max(
                    peak_rss_mb,
                    process.memory_info().rss / (1024**2),
                )

        validation_result = evaluate_model(model, validation_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(1, len(train_targets)),
            "train_accuracy": accuracy_score(train_targets, train_predictions),
            "train_f1_macro": f1_score(
                train_targets, train_predictions, average="macro", zero_division=0
            ),
            "validation_loss": validation_result["metrics"]["loss"],
            "validation_accuracy": validation_result["metrics"]["accuracy"],
            "validation_f1_macro": validation_result["metrics"]["f1_macro"],
        }
        history.append(row)
        print(
            f"Epoch {epoch}/{epochs}: train_loss={row['train_loss']:.4f}, "
            f"validation_loss={row['validation_loss']:.4f}, "
            f"validation_f1_macro={row['validation_f1_macro']:.4f}"
        )
        if row["validation_loss"] < best_loss:
            best_loss = row["validation_loss"]
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    training_seconds = time.perf_counter() - started
    if process is not None:
        peak_rss_mb = max(peak_rss_mb, process.memory_info().rss / (1024**2))
    return {
        "model": model,
        "history": pd.DataFrame(history),
        "training_seconds": training_seconds,
        "peak_rss_mb": peak_rss_mb,
        "start_rss_mb": start_rss_mb,
        "parameters": parameter_counts(model)[0],
        "trainable_parameters": parameter_counts(model)[1],
    }


def stratified_nested_subsets(labels, fractions, seed: int = RANDOM_SEED):
    """Tạo các tập con phân tầng lồng nhau để so sánh lượng dữ liệu."""
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    by_class = {}
    for class_id in np.unique(labels):
        indices = np.flatnonzero(labels == class_id)
        rng.shuffle(indices)
        by_class[class_id] = indices
    subsets = {}
    for fraction in fractions:
        parts = [indices[: int(round(len(indices) * fraction))] for indices in by_class.values()]
        subsets[fraction] = np.sort(np.concatenate(parts))
    return subsets
