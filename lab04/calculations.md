LAB 04 — Tính toán trước khi chạy code

Bài 1 — Binary classification

x = [1, 2, 0]
w = [0.5, -0.2, 0.4]
b = 0.1

z = wᵀx + b
  = ______________________________________________________

σ(z) = 1 / (1 + e^(-z)) = _______________________________
P(y = 1 | x) = ___________________________________________

Bài 2 — Prediction threshold

P(y = 1 | x) = 0.72

Threshold 0.5: dự đoán ____________________________________
Threshold 0.8: dự đoán ____________________________________

Khi tăng threshold, precision thay đổi thế nào? ___________
Recall thay đổi thế nào? __________________________________

Bài 3 — Confusion matrix

                      Predicted positive   Predicted negative
Actual positive             80                    20
Actual negative             10                    90

TP = ______  FP = ______  FN = ______  TN = ______
Precision = TP / (TP + FP) = _____________________________
Recall    = TP / (TP + FN) = _____________________________
F1        = 2 × Precision × Recall / (Precision + Recall)
          = ______________________________________________

Nếu đây là bài toán phát hiện spam:
False positive nghĩa là __________________________________
False negative nghĩa là __________________________________

Bài 4 — Accuracy trap

950 negative, 50 positive; model dự đoán tất cả là negative.

Accuracy = _______________________________________________
Model có phát hiện được positive không? __________________
Ngoài accuracy cần xem metric nào? _______________________
