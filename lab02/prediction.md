1. Vocabulary không tăng khi chuyển unigram sang bigram, trigram. Dự đoán: đúng; vocabulary vẫn có 101.836 từ.

2. Số n-gram phân biệt tăng theo n vì context dài hơn tạo nhiều chuỗi riêng. Kết quả: unigram 101.837, bigram 1.232.091, trigram 2.630.557. Số unigram có tính cả ký hiệu kết thúc câu.

3. Trigram dễ gặp xác suất zero nhất, sau đó là bigram. Kết quả: MLE bigram và trigram đều có perplexity vô hạn trên validation/test vì gặp n-gram chưa thấy.

4. Trigram dự kiến có training perplexity thấp nhất. Kết quả MLE: unigram 1651, bigram 120, trigram 9.38.

5. Corpus nhỏ không đảm bảo trigram tốt hơn bigram. Kết quả: đúng; Laplace trigram có perplexity validation/test cao hơn Laplace bigram.

6. Với câu “the cat eats fish” và “the dog eats fish”, ban đầu đoán câu thứ nhất cao hơn. Tính theo bigram của corpus bài tập, hai câu bằng nhau: mỗi câu có xác suất 1/24.
