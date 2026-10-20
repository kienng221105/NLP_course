# Reflection — LAB 05

Dataset sau khi bỏ 422 review trùng: 49.578 mẫu, gần cân bằng giữa positive (50,19%) và negative (49,81%).

1. Hidden state được cập nhật theo từng token bằng thông tin token hiện tại và state trước đó, nên nó có thể mang ngữ cảnh từ các từ trước sang bước tiếp theo. Ví dụ, khi đọc “like”, state có thể đã chứa tín hiệu phủ định từ “not”. Tuy vậy, state là biểu diễn có kích thước cố định, nên thông tin xa vẫn có thể bị suy giảm.

2. Trên validation, GRU đạt F1 macro cao nhất trong các recurrent model (0,8162), tiếp theo là LSTM (0,7974) và RNN (0,7314). Kết quả này phù hợp với dự đoán rằng các gate giúp xử lý thông tin chuỗi tốt hơn RNN thường trong thí nghiệm này. Nhưng TF-IDF + Logistic Regression vẫn cao hơn (F1 macro 0,8940), nên kết quả không chứng minh recurrent model luôn tốt hơn.

3. Khi max_length tăng 50 → 100 → 200, tỷ lệ token được giữ tăng 21,53% → 41,59% → 68,70%, test F1 macro tăng 0,7654 → 0,8115 → 0,8466. Đổi lại, thời gian train tăng 57,5 → 157,9 → 793,8 giây. Chuỗi dài giữ thêm ngữ cảnh nhưng tốn thời gian đáng kể hơn.

4. Model tốt nhất tổng thể là TF-IDF + Logistic Regression: validation F1 macro 0,8940; test accuracy 0,8949 và test F1 macro 0,8949; train khoảng 6,14 giây với 82.646 feature. Trong các recurrent model, GRU tốt nhất ở validation (F1 macro 0,8162), nhưng cần khoảng 157,9 giây và có 1.945.025 tham số. Vì vậy lựa chọn phụ thuộc việc ưu tiên hiệu quả và tốc độ hay biểu diễn chuỗi.

5. RNN phải truyền thông tin qua từng timestep; thông tin ở xa cần đi qua nhiều phép cập nhật và gradient có thể suy giảm hoặc tăng quá mức. Attention tạo động lực cho cách truy cập trực tiếp các vị trí quan trọng trong chuỗi, thay vì chỉ dựa vào một đường truyền recurrent dài. Thí nghiệm vị trí từ trong lab này chỉ là quan sát minh họa, không chứng minh vanishing gradient.
