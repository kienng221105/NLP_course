# Prediction trước thí nghiệm

Dataset: IMDb — phân loại review thành positive/negative.

## 1. TF-IDF + Logistic Regression hay RNN + Embedding xử lý word order trực tiếp hơn?

Prediction: RNN + Embedding.

Reason: RNN đọc token lần lượt và cập nhật hidden state theo token trước đó. TF-IDF unigram biểu diễn trọng số từ nhưng không giữ thứ tự câu.

Confidence: Cao.

## 2. RNN hay LSTM xử lý dependency dài tốt hơn?

Prediction: LSTM có khả năng xử lý dependency dài tốt hơn RNN thường.

Reason: Cell state và các gate của LSTM điều tiết việc giữ, cập nhật thông tin qua nhiều timestep, giúp giảm khó khăn khi truyền thông tin xa.

Confidence: Khá cao, kết quả còn phụ thuộc dữ liệu và cách train.

## 3. Khi sequence length tăng từ 50 lên 200, training thay đổi thế nào?

Prediction: Training sẽ lâu hơn và dùng thêm bộ nhớ; model có thêm cơ hội đọc ngữ cảnh xa hơn. F1 có thể tăng nếu phần cuối chứa thông tin hữu ích, nhưng không chắc tăng.

Reason: RNN phải xử lý nhiều timestep hơn. Chuỗi dài hơn cũng giữ lại nhiều token hơn nhưng làm chi phí tính toán tăng.

Confidence: Cao về thời gian, vừa phải về F1.

## 4. Truncate review về 30 token có thể làm mất thông tin gì?

Prediction: Có thể mất phần đánh giá quan trọng nằm sau 30 token, như kết luận, phủ định hoặc ý chuyển hướng sau “but”.

Reason: Notebook giữ phần đầu chuỗi rồi cắt phần còn lại; nếu sentiment chính nằm ở cuối review, model không được nhìn thấy tín hiệu đó.

Confidence: Cao (0,9).

## 5. LSTM có luôn đạt accuracy cao hơn RNN không?

Prediction: Không.

Reason: LSTM có cơ chế mạnh hơn cho dependency dài nhưng cũng có nhiều tham số và chi phí train hơn. Nếu dữ liệu hoặc chuỗi không cần thông tin xa, lợi thế đó có thể không làm accuracy tăng.

Confidence: Cao (0,9).

## Đối chiếu sau khi chạy

1. RNN xử lý chuỗi theo thứ tự, nhưng validation F1 macro của RNN (0,7314) thấp hơn TF-IDF + Logistic Regression (0,8940). Khả năng dùng thứ tự không đồng nghĩa kết quả luôn cao hơn.

2. LSTM đạt validation F1 macro 0,7974, cao hơn RNN (0,7314); GRU cao nhất trong nhóm recurrent với 0,8162. Kết quả phù hợp dự đoán cho LSTM trong lần chạy này.

3. Khi max_length tăng 50 → 200, thời gian train tăng 57,5 → 793,8 giây. Test F1 macro cũng tăng 0,7654 → 0,8466 trong các giá trị đã thử.

4. Thí nghiệm không chạy max_length=30 nên chưa đo trực tiếp mức ảnh hưởng đó. Review có sentiment nằm sau vị trí cắt có thể mất tín hiệu; ở application, câu contrast có ý tích cực vẫn bị dự đoán negative.

5. LSTM không luôn đứng đầu: lần chạy này LSTM tốt hơn RNN, nhưng GRU tốt hơn LSTM và TF-IDF vẫn là model tốt nhất tổng thể. Kết quả trên một dataset không đảm bảo thứ hạng này đúng cho dataset khác.
