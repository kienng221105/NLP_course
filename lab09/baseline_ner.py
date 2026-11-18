"""Word-feature Logistic Regression baseline cho CoNLL-2003 NER."""

from __future__ import annotations

from typing import Iterable

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from information_extraction import entity_metrics


def _tag_name(tag, label_names=None) -> str:
    if isinstance(tag, int) and label_names is not None:
        return label_names[tag]
    return str(tag)


def word_shape(word: str) -> str:
    """Mẫu hình chữ hoa/chữ thường/số của một token."""
    result = []
    for character in word:
        if character.isupper():
            result.append("X")
        elif character.islower():
            result.append("x")
        elif character.isdigit():
            result.append("d")
        else:
            result.append(character)
    return "".join(result)


def token_features(words: list[str], index: int) -> dict[str, object]:
    """Token hiện tại, prefix/suffix và context window ±2."""
    word = words[index]
    features = {
        "bias": 1.0,
        "word.lower": word.lower(),
        "word.isupper": word.isupper(),
        "word.istitle": word.istitle(),
        "word.isdigit": word.isdigit(),
        "word.shape": word_shape(word),
    }
    for size in (1, 2, 3):
        features[f"prefix{size}"] = word[:size].lower()
        features[f"suffix{size}"] = word[-size:].lower()
    for offset in (-2, -1, 1, 2):
        neighbor = index + offset
        if 0 <= neighbor < len(words):
            value = words[neighbor]
            features[f"{offset}.lower"] = value.lower()
            features[f"{offset}.istitle"] = value.istitle()
            features[f"{offset}.isupper"] = value.isupper()
        else:
            features[f"{offset}.boundary"] = True
    return features


def sentence_features(words: list[str]) -> list[dict[str, object]]:
    return [token_features(words, i) for i in range(len(words))]


def train_baseline(train_rows: Iterable[dict], label_names=None, random_state: int = 42):
    """Huấn luyện Logistic Regression trên word-context features."""
    feature_rows = []
    labels = []
    for row in train_rows:
        feature_rows.extend(sentence_features(row["tokens"]))
        labels.extend(_tag_name(tag, label_names) for tag in row["ner_tags"])

    vectorizer = DictVectorizer(sparse=True)
    x_train = vectorizer.fit_transform(feature_rows)
    classifier = LogisticRegression(
        max_iter=150,
        solver="lbfgs",
        class_weight="balanced",
        random_state=random_state,
    )
    classifier.fit(x_train, labels)
    return {"vectorizer": vectorizer, "classifier": classifier}


def predict_baseline(model, rows: Iterable[dict]) -> list[list[str]]:
    predictions = []
    for row in rows:
        features = sentence_features(row["tokens"])
        x = model["vectorizer"].transform(features)
        predictions.append(model["classifier"].predict(x).tolist())
    return predictions


def evaluate_baseline(model, rows: list[dict], label_names=None):
    rows = list(rows)
    predicted = predict_baseline(model, rows)
    gold_tags = [[_tag_name(tag, label_names) for tag in row["ner_tags"]] for row in rows]
    words = [row["tokens"] for row in rows]
    return entity_metrics(words, gold_tags, predicted), predicted
