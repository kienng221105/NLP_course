Dataset/task: IMDb.

1. TF-IDF unigram + Linear SVM phù hợp nhất trong các mô hình đã thử: validation macro F1 = 0,8967; test macro F1 = 0,8945.

2. Average Word2Vec đạt validation macro F1 = 0,8589, thấp hơn TF-IDF. Vector trung bình có thể làm mờ từ/cụm từ cảm xúc đặc trưng.

3. Khi class imbalance, accuracy có thể cao dù mô hình bỏ sót lớp ít mẫu. Vì vậy cần xem thêm precision, recall, F1 và confusion matrix. Dataset lần chạy này tương đối cân bằng.

4. Average Word2Vec lấy trung bình vector từ nên không giữ thứ tự. Cosine giữa “good” và “not good” là 0,975, cho thấy hai biểu diễn rất giống nhau; từ “not” chưa được thể hiện rõ.

5. Bigram giúp biểu diễn cụm từ như “not good”. F1 macro validation tăng nhẹ 0,8940 → 0,8956, nhưng vocabulary tăng 82.646 → 1.730.488 feature và thời gian train tăng 3,32 → 21,80 giây.

6. Confusion matrix trên test có 572 negative bị dự đoán thành positive và 474 positive bị dự đoán thành negative. Hai loại lỗi khá gần nhau; false positive nhiều hơn 98 mẫu.

7. Những hạn chế về thứ tự từ, phủ định và ngữ cảnh cho thấy hướng tiếp theo nên thử mô hình giữ thông tin chuỗi/ngữ cảnh, rồi so sánh với baseline hiện tại trên cùng cách chia dữ liệu.
