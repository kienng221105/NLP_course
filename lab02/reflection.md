# Reflection

1.Context dài hơn: bigram dùng một từ trước, trigram dùng hai từ trước để ước lượng từ tiếp theo.

2.Số chuỗi có thể có tăng nhanh, trong khi corpus chỉ quan sát được một phần nhỏ; nhiều context/ngram chỉ xuất hiện một lần hoặc chưa xuất hiện.

3.MLE gán xác suất 0 cho n-gram chưa thấy, làm xác suất cả câu bằng 0 và perplexity vô hạn. Smoothing cấp xác suất dương cho các trường hợp đó, dù cách Laplace đơn giản có thể làm xác suất bị dàn quá rộng khi vocabulary lớn.

4.Mức độ bất ngờ/bối rối trung bình của mô hình khi dự đoán token trong dữ liệu đánh giá. Cần so sánh cùng dữ liệu và cùng quy ước tiền xử lý.

5.Không. Perplexity đánh giá khả năng dự đoán token theo phân phối dữ liệu; nó không trực tiếp đo tính đúng sự thật, mạch lạc dài hạn, phù hợp ngữ cảnh hay chất lượng văn phong.

6.Nó chỉ nhớ context ngắn, phụ thuộc mạnh vào tần suất corpus, khó xử lý quan hệ xa, nghĩa, kiến thức thế giới, cấu trúc mới và các biến thể chưa từng thấy.

7.Không. Trigram chỉ dùng tối đa hai token ngay trước từ cần dự đoán; các từ xa hơn bị bỏ qua.
