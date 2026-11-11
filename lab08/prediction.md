# Prediction — LAB 08

Các dự đoán dưới đây được ghi trước khi chạy các experiment.

## Prediction 1 — Pretrained so với from scratch

**Prediction:** BERT pretrained sau fine-tuning sẽ đạt F1 cao hơn model cùng kiến trúc khởi tạo ngẫu nhiên.

**Reason:** Pretraining cung cấp biểu diễn ngôn ngữ ban đầu; model from scratch phải học cả biểu diễn lẫn ranh giới phân loại chỉ từ nhãn IMDb.

**Expected evidence:** Accuracy và F1 trên cùng test split, tokenizer, maximum sequence length, batch size, optimizer và số epoch.

## Prediction 2 — Frozen encoder so với fine-tuning

**Prediction:** Frozen encoder vẫn phân loại được nhờ biểu diễn pretrained, nhưng dự kiến thấp hơn fine-tuning toàn bộ model.

**Reason:** Head có thể học cách dùng feature sẵn có; fine-tuning còn điều chỉnh encoder cho task sentiment IMDb.

**Expected evidence:** Accuracy/F1, số tham số trainable và thời gian train trên cùng split.

## Prediction 3 — Ít training data

**Prediction:** Với 10% train data, BERT from scratch sẽ có F1 thấp hơn pretrained model.

**Reason:** Model from scratch ít nhãn để học representation và task; pretrained model đã có representation ban đầu.

**Expected evidence:** Validation F1 macro của hai model tại cùng 10% train subset, cùng tokenizer, split, sequence length, batch size và epoch.

## Đối chiếu sau khi chạy

| Prediction | Kết quả đo được | Đối chiếu |
|---|---|---|
| Pretrained > from scratch | Test F1 macro: `0.6950` so với `0.3342`; accuracy `0.7033` so với `0.5019` | Được ủng hộ trong cấu hình chạy |
| Fine-tuning > frozen | Test F1 macro: `0.6950` so với `0.5559`; frozen chỉ train 258 tham số | Được ủng hộ |
| Pretrained giữ kết quả tốt hơn ở 10% | Validation F1 macro: pretrained `0.5995`, scratch `0.3342` trên cùng 2.975 review | Được ủng hộ trong lần chạy này |

Các so sánh này dùng BERT nhỏ và một epoch cho model chính; kết quả mô tả cấu hình của lab, không đảm bảo sẽ giữ nguyên với model, epoch hay dataset khác.
