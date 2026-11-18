# Prediction — LAB 09

Các dự đoán ban đầu được giữ ở đầu từng mục; bảng sau cùng đối chiếu với số liệu notebook.

## Prediction 1 — Baseline vs pretrained Transformer

**Prediction:** pretrained Transformer sẽ đạt entity-level F1 cao hơn baseline word-feature Logistic Regression.

**Reason:** contextual representation pretrained có thể tận dụng context rộng hơn các feature cục bộ.

**Expected evidence:** entity-level precision, recall, F1 trên cùng CoNLL-2003 test split.

## Prediction 2 — Độ dài entity

**Prediction:** entity nhiều token sẽ khó nhận diện hoàn chỉnh hơn entity một token.

**Reason:** exact match cần đúng toàn bộ boundary và loại entity; lỗi một token làm cả span sai.

**Expected evidence:** entity-level F1, support và số lỗi theo nhóm 1, 2, 3, 4+ tokens.

## Prediction 3 — Rare entities

**Prediction:** entity hiếm sẽ có recall/F1 thấp hơn vì ít ví dụ train và nhiều tên chưa gặp.

**Reason:** model có ít pattern để học, đặc biệt nếu tên hiếm bị tách thành subword.

**Expected evidence:** per-type support/F1 và các lỗi rare entity trong test.

## Đối chiếu sau khi chạy

| Prediction | Kết quả đo được | Đánh giá |
|---|---|---|
| Pretrained > baseline | Baseline test entity F1 `0,7334`; Transformer `0,6463` | Không được ủng hộ ở cấu hình này. Baseline có capitalization và context ±2; BERT nhỏ là uncased. |
| Entity dài khó hơn | F1: 1 token `0,6420`; 2 `0,7017`; 3 `0,3966`; 4+ `0,0533` | Được ủng hộ với entity 3+ tokens; nhóm 2-token cao hơn nhóm 1-token. Support nhóm dài nhỏ hơn. |
| Rare entity dễ lỗi hơn | MISC support 702/F1 `0,5007`; PER support 1.616/F1 `0,7691`; LOC support 1.667/F1 `0,7300`; ORG support 1.661/F1 `0,4902` | Một phần được ủng hộ; ORG có support cao nhưng F1 thấp, nên tần suất không phải yếu tố duy nhất. |
