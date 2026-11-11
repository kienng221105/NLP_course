# LAB 08 — Pretrained Transformers và BERT Fine-Tuning

## Bài toán và dữ liệu

Phân loại cảm xúc positive/negative. Tập chính là IMDb tại `../data/IMDB Dataset.csv`; domain shift dùng product reviews trong Amazon Polarity.

## Cấu trúc bài nộp

- `README.md` — cấu hình tái lập và cách chạy.
- `calculations.md` — phép tính MLM, tokenization và câu hỏi ôn tập.
- `prediction.md` — dự đoán trước experiment và đối chiếu sau khi chạy.
- `tokenization.ipynb` — quan sát subword tokens, IDs, special tokens và attention mask.
- `pretrained_classification.py` — hàm xử lý dữ liệu, tokenizer, model và huấn luyện.
- `fine_tuning.ipynb` — các experiment và kết quả chạy.
- `results.csv` — metric theo experiment, split, số mẫu và cấu hình.
- `data_efficiency.png` — validation F1 theo lượng dữ liệu, có điểm scratch 10%.
- `sequence_length.png` — F1 và thời gian theo độ dài sequence.
- `confusion_matrix.png` — nhầm lẫn trên IMDb test sample.
- `confidence_analysis.png` — confidence của dự đoán đúng/sai.
- `error_analysis.md` — ví dụ lỗi và câu hỏi phân tích.
- `reflection.md` — reflection sau lab.

## Cấu hình tái lập

IMDb có 50.000 dòng ban đầu; sau khi bỏ review trùng còn 49.578 dòng. Chia stratified 60/20/20 với seed 42: train 29.746, validation 9.916, test 9.916. Model/tokenizer: `google/bert_uncased_L-2_H-128_A-2`; BERT có 4.386.178 tham số và hidden size 128.

Experiment 1–3 dùng max_length 128, batch size 128, AdamW, learning rate `2e-5`, một epoch. Sequence sweep dùng cùng subset 25% (7.436 review), batch 32, một epoch và cùng 2.000 mẫu validation/test. Domain shift và error/confidence analysis dùng model 3 epoch trên cùng 25% subset; IMDb test gồm 2.000 review, Amazon test gồm 1.000 review cân bằng lấy từ `test[:2000]`.

Môi trường chạy: Python 3.11, PyTorch 2.14.1+cpu, Transformers 5.22.2, Datasets 5.2.0; CPU, không có CUDA. Các cell huấn luyện giới hạn một thread. Thời gian train được lưu riêng trong `results.csv`; thời gian phụ thuộc máy. Khi nạp encoder pretrained cho classification, classification head được khởi tạo mới để học nhãn sentiment.

## Chạy notebook

Cài thư viện bằng `python -m pip install torch transformers datasets pandas scikit-learn matplotlib psutil`. Dùng Jupyter hoặc VS Code để mở notebook. Mở `tokenization.ipynb`, rồi chạy `fine_tuning.ipynb` từ trên xuống. Lần đầu cần mạng để tải model/tokenizer và Amazon Polarity; những lần sau có thể dùng cache.
