# Reflection

Dự đoán vocabulary khoảng 100.000–200.000 từ, thấp hơn kết quả thực tế của Pipeline A là 263.260 từ. Số từ tăng nhiều do corpus có URL, tên riêng, lỗi chính tả, ký tự Unicode và các từ đi kèm dấu câu.

Dự đoán về sparsity phù hợp với kết quả thực nghiệm. Cả ba pipeline đều có hơn 99% số 0. Mỗi tài liệu chỉ chứa một phần rất nhỏ vocabulary của toàn bộ corpus.

Một điểm đáng chú ý là các token có IDF cao nhất không phải lúc nào cũng có ích. Danh sách này có nhiều mã số, ký hiệu và chuỗi ký tự lỗi. Những token đó hiếm vì chỉ xuất hiện trong rất ít tài liệu, nhưng không mang nhiều ý nghĩa cho việc tìm kiếm.

Trên candidate pool và các nhãn relevance hiện có, Pipeline A và B cho kết quả tốt nhất với P@5 = 0,60, Recall@5 = 0,9286 và MRR = 0,875 (Pipeline A) / 0,833 (Pipeline B). Pipeline C có cùng P@5 nhưng Recall@5 thấp hơn một chút, bằng 0,8786. Các query không có document relevant trong candidate pool được dùng để chẩn đoán lỗi và không được đưa vào metric tổng hợp. Kết quả này phụ thuộc vào các nhãn relevance hiện có và cho thấy vocabulary nhỏ hơn không đồng nghĩa với tìm kiếm tốt hơn.

Query `transformer language model` thể hiện rõ giới hạn của TF-IDF. Kết quả đứng đầu nói về biến áp điện, trong khi query nói về mô hình ngôn ngữ. Hệ thống nhận ra các từ trùng nhau nhưng không hiểu nghĩa của từ trong ngữ cảnh. Tương tự, "heart attack treatment" và "myocardial infarction therapy" cùng nghĩa y khoa nhưng không chia sẻ token nào, dẫn đến hai tập kết quả rời nhau hoàn toàn.

Bài lab cho thấy preprocessing làm thay đổi vocabulary, vector TF-IDF và thứ tự kết quả tìm kiếm. TF-IDF hữu ích khi các tài liệu có từ trùng với query, nhưng còn hạn chế khi cần hiểu nghĩa và ngữ cảnh.
