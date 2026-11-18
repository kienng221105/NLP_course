# LAB 09 — Named Entity Recognition and Information Extraction

## Dataset và task

CoNLL-2003 tiếng Anh, tải từ Hugging Face dataset `lhoestq/conll2003`. Splits: 14.041 câu train, 3.250 validation, 3.453 test. Vì mirror trả `ner_tags` dưới dạng số nguyên không có `ClassLabel.names`, notebook ánh xạ theo thứ tự đề: `O`, `B/I-PER`, `B/I-ORG`, `B/I-LOC`, `B/I-MISC`.

## Nội dung

- `tokenizer_alignment.ipynb`: thống kê tokenization trên 20 câu, `word_ids`, first-subword alignment và `-100`.
- `baseline_ner.py`: Logistic Regression với token, shape, prefix/suffix, capitalization và context ±2.
- `ner_finetuning.ipynb`: baseline, fine-tuning token classifier, token/entity metrics, type/length analysis, context cases và IE.
- `information_extraction.py`: BIO span parsing, exact entity metrics và group entities thành structured data. Mini IE chọn Option A — News, chỉ đánh giá PER/ORG/LOC trên 30 câu test.
- `results.csv`, `entity_metrics.csv`, `entity_length_analysis.png`, `confusion_matrix.png`, `error_analysis.md`, `calculations.md`, `prediction.md`, `reflection.md` chứa kết quả và bài báo cáo.

## Cấu hình tái lập

| Thuộc tính | Giá trị |
|---|---|
| Model/tokenizer | `google/bert_uncased_L-2_H-128_A-2` |
| Encoder | 2 layers, hidden size 128 |
| Parameter count | 4.370.569 |
| Alignment | Nhãn ở subword đầu; special/continuation tokens `-100` |
| Max length | 128 |
| Batch size | 32 |
| Epochs | 5; best checkpoint theo validation entity F1 |
| Optimizer / learning rate | AdamW / `2e-5` |
| Seed / device | 42 / CPU một thread |
| Baseline train time | 34,0 giây |
| Transformer train time | 239,2 giây |

Test entity-level F1: baseline `0,7334`; Transformer `0,6463`. Test token accuracy: baseline `0,9507`; Transformer `0,9366`. Token accuracy cao hơn entity F1 vì entity cần đúng đồng thời nhãn và boundary. Mini Information Extraction trên 30 câu test đạt exact-entity precision `0,6923`, recall `0,6792`, F1 `0,6857`.

## Chạy

Cài `torch`, `transformers`, `datasets`, `scikit-learn`, `pandas` và `matplotlib`. Chạy `tokenizer_alignment.ipynb` rồi `ner_finetuning.ipynb` từ trên xuống. Lần đầu cần internet để tải dataset/model; sau đó các file được cache.
