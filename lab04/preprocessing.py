"""Chuẩn hóa văn bản và chia dữ liệu có stratification cho LAB 04."""

import re
from collections.abc import Sequence

from sklearn.model_selection import train_test_split


def normalize_text(text: str) -> str:
    """Chuẩn hóa chữ hoa và khoảng trắng; giữ nguyên từ phủ định như 'not'."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    return re.sub(r"\s+", " ", text.strip().lower())


def split_dataset(
    texts: Sequence[str],
    labels: Sequence[str],
    test_size: float = 0.2,
    validation_size: float = 0.1,
    random_state: int = 42,
):
    """Chia train/validation/test; validation_size tính trên toàn dataset."""
    if len(texts) != len(labels):
        raise ValueError("Số văn bản phải bằng số nhãn")
    if not texts:
        raise ValueError("Dataset rỗng")
    if test_size <= 0 or validation_size <= 0:
        raise ValueError("test_size và validation_size phải lớn hơn 0")
    if test_size + validation_size >= 1:
        raise ValueError("Tổng test_size và validation_size phải nhỏ hơn 1")

    x_train_val, x_test, y_train_val, y_test = train_test_split(
        list(texts),
        list(labels),
        test_size=test_size,
        random_state=random_state,
        stratify=list(labels),
    )
    relative_validation_size = validation_size / (1 - test_size)
    x_train, x_validation, y_train, y_validation = train_test_split(
        x_train_val,
        y_train_val,
        test_size=relative_validation_size,
        random_state=random_state,
        stratify=y_train_val,
    )
    return x_train, x_validation, x_test, y_train, y_validation, y_test
