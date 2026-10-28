# Reflection — LAB 06

1. RNN/LSTM truyền thông tin qua hidden state theo từng bước nên token xa phải đi qua nhiều timestep. Self-attention cho mỗi token tính trọng số với toàn bộ token khác, giúp truy cập ngữ cảnh xa trực tiếp hơn.

2. Query biểu diễn thông tin đang cần tìm; Key dùng để tính mức phù hợp với Query; Value là nội dung được tổng hợp. Softmax của scores tạo weights, rồi weighted sum các Value thành context vector.

3. Scaling chia scores cho căn bậc hai của key dimension, làm softmax bớt quá tập trung. Trong thí nghiệm temperature, max weight giảm từ 0,867 ở T=0,5 xuống 0,506 ở T=2; entropy tăng từ 0,441 lên 1,020, distribution phẳng hơn.

4. Heatmap cho thấy model phân phối attention weights giữa các token. Nó không chứng minh model hiểu ngữ pháp hay reasoning; ví dụ câu “I like this movie” bị dự đoán negative dù attention có weight cao cho like.

5. Attention đạt test accuracy 0,8200 và macro F1 0,8198; LSTM đạt lần lượt 0,7936 và 0,7936. Attention train khoảng 31,75 giây với 1.932.353 tham số, còn LSTM mất 327,67 giây với 1.953.345 tham số. Attention tốt hơn trong lần chạy này, nhưng kết quả không đảm bảo sẽ như vậy trên dataset khác.

6. Với T token, full self-attention tạo matrix T×T nên số phần tử tăng theo T². Ở T=10.000, riêng một matrix float32 đã khoảng 381 MiB, chưa tính batch, gradients và activations. Vì vậy bước tiếp theo cần tìm cách xây dựng attention hiệu quả hơn cho chuỗi dài.
