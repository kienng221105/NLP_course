# Prediction 1 — Vocabulary

Với corpus khoảng 30.000 documents, em dự đoán vocabulary sẽ có kích thước khoảng 100.000 đến 200.000. Corpus được thu thập từ nhiều nguồn web khác nhau nên có thể chứa tên riêng, số, từ hiếm, từ viết sai và nhiều dạng biểu diễn khác nhau của cùng một khái niệm.

Em dự đoán một phần đáng kể vocabulary chỉ xuất hiện trong một hoặc một vài documents.

# Prediction 2 — Sparsity

Mỗi document chỉ sử dụng một phần rất nhỏ vocabulary của toàn corpus, trong khi vector của document có chiều bằng toàn bộ vocabulary.

Vì vậy, em dự đoán tỷ lệ zero entries sẽ rất cao, có thể trên 97%.

# Prediction 3 — Search

Với một query bất kỳ, em dự đoán các documents đứng đầu kết quả tìm kiếm không nhất thiết là những documents gần nghĩa nhất, tfidf và cosine similarity chủ yếu dựa trên lexical overlap giữa query và document.

Một document chứa nhiều từ giống query có thể được xếp hạng cao dù nội dung chưa chắc phù hợp hoàn toàn. Ngược lại, một document gần nghĩa nhưng dùng từ đồng nghĩa hoặc cách diễn đạt khác có thể bị xếp hạng thấp vì không có đủ token trùng nhau.
