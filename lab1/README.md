# Lab 01 — Từ xử lý văn bản đến tìm kiếm

## Tệp bài làm

- `calculations.md`: phiếu bài tập tính tay Part B; tự hoàn thành trước khi kiểm chứng bằng code.
- `prediction.md`: ba dự đoán Part C; cần bảo đảm là dự đoán của sinh viên trước khi xem kết quả.
- `implementation.py`: cài đặt TF-IDF và cosine similarity cho Part E.
- `experiments.ipynb`: kiểm tra corpus 30.000 documents, ablation preprocessing và search.
- `evaluation_set.csv`: 8 query; tự điền ID tài liệu liên quan, cách nhau bằng dấu `;`.
- `results.csv`: top-5 kết quả cho mỗi query và pipeline, được tạo khi chạy phần search.
- `evaluation_metrics.csv`: Precision@5, Recall@5 và MRR, được tạo sau khi đã gán nhãn.
- `reflection.md`: learning check và reflection cuối buổi.
- `AI_contribution.md`: khai báo việc sử dụng AI.
- `W1.pdf`: đề bài.

## Chạy notebook

Mở notebook với kernel Python có `pandas`, `numpy`, `scikit-learn`, `nltk` và `transformers`. Pipeline C dùng tokenizer WordPiece `bert-base-uncased`; lần đầu chạy cần tải tokenizer từ Hugging Face và cần có kết nối mạng. Dataset được đọc tương đối từ `../data/c4-train.00000-of-01024-30K.json.gz`, vì vậy chạy notebook với working directory là thư mục `lab1`.

Chạy các cell theo thứ tự. Sau Part D, cell đầu Part F giải phóng các ma trận và token list lớn để giảm áp lực RAM. Pipeline F đo trên cùng phép chia train/test 80/20 với seed cố định. Phần search dựng lại mỗi index theo từng pipeline để tránh giữ ba ma trận TF-IDF cùng lúc.

Trước khi chạy cell tính metric, xem top kết quả trong `results.csv` và tự gán relevance vào `evaluation_set.csv`. Không điền nhãn dựa chỉ trên thứ hạng của hệ thống.
