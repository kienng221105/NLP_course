LAB 03 — Tính toán trước khi chạy code

Bài 1 — Co-occurrence, window k = 1

Corpus:
the cat eats fish
the dog eats fish
the cat likes milk
the dog likes meat

Thứ tự vocabulary: cat, dog, eats, likes, fish, milk, meat.
Từ `the` không thuộc vocabulary nên không được tính làm context.

cat:   [0, 0, 1, 1, 0, 0, 0]
dog:   [0, 0, 1, 1, 0, 0, 0]
eats:  [1, 1, 0, 0, 2, 0, 0]
likes: [1, 1, 0, 0, 0, 1, 1]

Bài 2 — Cosine similarity

x = [1, 2, 1], y = [2, 4, 2]

cos(x, y) = (x · y) / (||x|| ||y||)
          = 12 / (√6 × √24) = 12 / 12 = 1.
Ý nghĩa: y = 2x nên hai vector cùng hướng. Cosine similarity bằng 1 dù độ dài khác nhau.

Bài 3 — Semantic similarity

v_doctor = [0.8, 0.1, 0.7]
v_physician = [0.7, 0.2, 0.8]
v_banana = [-0.2, 0.9, -0.1]

Dự đoán trước khi tính: physician gần doctor hơn banana.
cos(doctor, physician) = 1.14 / (√1.14 × √1.17) ≈ 0.9871
cos(doctor, banana) = -0.14 / (√1.14 × √0.86) ≈ -0.1414
Kết luận: doctor và physician gần nhau hơn nhiều; banana gần như ngược hướng với doctor.

Bài 4 — Sparse và dense

1. Biểu diễn sparse: word-context vector 10.000 chiều, chỉ có 30 giá trị khác 0.
2. Biểu diễn dense: embedding 300 chiều, phần lớn giá trị khác 0.
3. Dense representation có thể hữu ích cho similarity vì nhiều chiều có thể cùng mã hóa
   đặc trưng ngữ nghĩa, giúp từ xuất hiện trong context tương tự có vector gần nhau.
4. Dense representation không luôn tốt hơn. Hiệu quả phụ thuộc vào corpus và tác vụ;
   sparse count dễ diễn giải và có thể hữu ích cho lexical matching chính xác.

Bài 5 — Training pairs, window = 1

Câu: the cat eats fish

CBOW (context → target):
____________________________________________________________

Skip-gram (target → context):
____________________________________________________________

Bài 6 — Analogy vector

king - man + woman = [8, 2, 7] - [5, 1, 5] + [5, 3, 5]

Vector kết quả: ____________________________________________
Quan hệ có thể biểu diễn: __________________________________
