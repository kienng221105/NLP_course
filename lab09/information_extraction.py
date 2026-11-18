"""Các hàm chuyển BIO tags thành spans và structured entity records."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

ENTITY_TYPES = ("PER", "ORG", "LOC", "MISC")


def bio_to_spans(words: list[str], tags: list[str]) -> list[dict]:
    """Chuyển một câu BIO thành entity spans có end exclusive."""
    if len(words) != len(tags):
        raise ValueError("Số words và BIO tags phải bằng nhau.")

    spans = []
    i = 0
    while i < len(tags):
        tag = tags[i]
        if tag == "O" or "-" not in tag:
            i += 1
            continue
        prefix, entity_type = tag.split("-", 1)
        if prefix not in {"B", "I"}:
            i += 1
            continue

        start = i
        i += 1
        while i < len(tags) and tags[i] == f"I-{entity_type}":
            i += 1
        spans.append({
            "start": start,
            "end": i,
            "type": entity_type,
            "text": " ".join(words[start:i]),
        })
    return spans


def entities_to_dict(entities: Iterable[dict]) -> dict[str, list[str]]:
    """Nhóm entity strings theo nhãn, giữ thứ tự xuất hiện."""
    grouped = defaultdict(list)
    for entity in entities:
        grouped[entity["type"]].append(entity["text"])
    return dict(grouped)


def _span_set(words_by_sentence: list[list[str]], tags_by_sentence: list[list[str]]):
    result = set()
    for sentence_id, (words, tags) in enumerate(zip(words_by_sentence, tags_by_sentence)):
        for entity in bio_to_spans(words, tags):
            result.add((sentence_id, entity["start"], entity["end"], entity["type"]))
    return result


def entity_metrics(
    gold_words: list[list[str]],
    gold_tags: list[list[str]],
    predicted_tags: list[list[str]],
    entity_types: Iterable[str] = ENTITY_TYPES,
) -> dict:
    """Tính exact-match entity-level micro và precision/recall/F1 theo type."""
    gold = _span_set(gold_words, gold_tags)
    predicted = _span_set(gold_words, predicted_tags)
    correct = gold & predicted

    def score(gold_set: set, predicted_set: set) -> dict[str, float | int]:
        tp = len(gold_set & predicted_set)
        fp = len(predicted_set - gold_set)
        fn = len(gold_set - predicted_set)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {"tp": tp, "fp": fp, "fn": fn, "support": tp + fn,
                "precision": precision, "recall": recall, "f1": f1}

    per_type = {}
    for entity_type in entity_types:
        gold_type = {span for span in gold if span[3] == entity_type}
        predicted_type = {span for span in predicted if span[3] == entity_type}
        per_type[entity_type] = score(gold_type, predicted_type)
    return {"overall": score(gold, predicted), "per_type": per_type,
            "gold_spans": gold, "predicted_spans": predicted,
            "correct_spans": correct}


def entity_length_metrics(entity_result: dict) -> dict[str, dict]:
    """Tính exact-match F1 theo độ dài entity gold: 1, 2, 3, 4+ tokens."""
    gold = entity_result["gold_spans"]
    predicted = entity_result["predicted_spans"]
    correct = entity_result["correct_spans"]

    def group(span):
        length = span[2] - span[1]
        return str(length) if length < 4 else "4+"

    output = {}
    for name in ("1", "2", "3", "4+"):
        gold_group = {span for span in gold if group(span) == name}
        pred_group = {span for span in predicted if group(span) == name}
        tp = len(correct & gold_group & pred_group)
        fp = len(pred_group - gold_group)
        fn = len(gold_group - pred_group)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        output[name] = {"support": len(gold_group), "tp": tp, "fp": fp,
                        "fn": fn, "precision": precision, "recall": recall, "f1": f1}
    return output


def find_boundary_errors(words_by_sentence, gold_tags, predicted_tags) -> list[dict]:
    """Lấy các gold/pred spans chồng lấn nhưng không khớp chính xác."""
    errors = []
    for sentence_id, (words, gold, predicted) in enumerate(
        zip(words_by_sentence, gold_tags, predicted_tags)
    ):
        gold_spans = bio_to_spans(words, gold)
        predicted_spans = bio_to_spans(words, predicted)
        gold_keys = {(x["start"], x["end"], x["type"]) for x in gold_spans}
        for pred in predicted_spans:
            key = (pred["start"], pred["end"], pred["type"])
            if key in gold_keys:
                continue
            for ref in gold_spans:
                overlaps = pred["start"] < ref["end"] and ref["start"] < pred["end"]
                same_boundary = pred["start"] == ref["start"] and pred["end"] == ref["end"]
                if overlaps and not same_boundary:
                    errors.append({"sentence_id": sentence_id, "words": words,
                                   "gold": ref, "predicted": pred})
    return errors


def evaluate_information_extraction(predicted_entities: list[list[dict]],
                                    gold_entities: list[list[dict]]) -> dict:
    """Tính micro exact-match precision/recall/F1 trên các sentence."""
    pred_set = set()
    gold_set = set()
    for sentence_id, entities in enumerate(predicted_entities):
        pred_set.update((sentence_id, e["start"], e["end"], e["type"]) for e in entities)
    for sentence_id, entities in enumerate(gold_entities):
        gold_set.update((sentence_id, e["start"], e["end"], e["type"]) for e in entities)
    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1}
