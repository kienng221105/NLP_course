1. TF-IDF + Logistic Regression: dự đoán hoạt động tốt vì review có nhiều từ cảm xúc đặc trưng. Kết quả: validation macro F1 = 0,8940.

2. Average Word2Vec không chắc tốt hơn TF-IDF vì phép lấy trung bình làm mất thứ tự và có thể làm mờ từ quan trọng. Kết quả: validation macro F1 = 0,8589, thấp hơn TF-IDF.

3. Vocabulary lớn hơn không đảm bảo accuracy tăng. Kết quả: Với bigram, vocabulary tăng từ 82.646 lên 1.730.488; validation macro F1 tăng nhẹ từ 0,8940 lên 0,8956.

4. Bỏ stopwords không chắc cải thiện kết quả vì một số từ như “not” có ý nghĩa với cảm xúc.

5. Accuracy và F1 có thể cho kết luận khác nhau, nhất là khi các lớp mất cân bằng. Kết quả: Linear SVM đạt validation accuracy 0,8967 và macro F1 0,8967.
