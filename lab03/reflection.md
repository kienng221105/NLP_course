| Representation | Context-dependent? | Sparse/Dense | Một từ có nhiều vector? |
|---|---|---|---|
| TF-IDF | Không | Sparse | Không |
| Co-occurrence | Không | Sparse | Không |
| Word2Vec | Không | Dense | Không |
| Contextual embedding | Có | Dense | Có |

Kết quả trên corpus

Corpus có 30.000 document, 634.941 câu, 10.972.699 token và 196.471 từ khác nhau trước min_count. Co-occurrence vocabulary còn 101.687 từ khi min_count = 2. Ma trận giữ cùng kích thước 101.687 × 101.687 ở các window; số non-zero tăng từ 4.835.875 (window 1) lên 9.305.554 (window 2) và 19.387.434 (window 5).

Với co-occurrence, cosine doctor-physician lần lượt là 0.913, 0.928, 0.947 khi window tăng 1, 2, 5. Doctor-hospital tăng 0.651 → 0.752 → 0.840; cat-dog tăng 0.869 → 0.918 → 0.955. Đây là kết quả của corpus này, không chứng minh window rộng hơn luôn tốt hơn.

Trong Word2Vec, doctor-physician lần lượt là 0.771, 0.764, 0.682 với window 2, 5, 10. Doctor-hospital là 0.497, 0.577, 0.562; cat-dog là 0.730, 0.807, 0.792. Ảnh hưởng của window khác nhau tùy cặp từ.

Word2Vec Skip-gram cho doctor-physician cosine 0.764, cat-dog 0.807, king-queen 0.735, car-automobile 0.734 và computer-banana 0.105. Analogy king - man + woman trả queen ở top 1. Kết quả cho thấy vector có thể phản ánh pattern trong corpus, không phải bằng chứng rằng model hiểu quan hệ như con người.

Dimension 50, 100, 300 lần lượt cho doctor-physician cosine 0.839, 0.764, 0.586; kích thước mảng vector tăng từ 38.8 MB lên 77.6 MB và 232.7 MB. Điểm cosine cao nhất cho truy vấn semantic search medical treatment lần lượt là 0.873, 0.828, 0.722; đây chỉ là similarity score, không có nhãn để xác nhận kết quả đúng. Analogy top-1 đều là queen. Dimension lớn hơn không đảm bảo similarity tốt hơn. Skip-gram mất khoảng 954 giây, so với 397 giây của CBOW trong lần chạy này.

Tại sao bank cần contextual representation?

Word2Vec tĩnh chỉ có một vector cho bank. Trong kết quả chạy, lookup bank ở câu về gửi tiền và câu về bờ sông cho cùng vector (khoảng cách 0). Contextual embedding dùng các từ xung quanh nên có thể tạo vector khác nhau cho hai cách dùng.

Ghi chú: cell most_similar của co-occurrence trong lần chạy cũ có cảnh báo RuntimeWarning khi tính chuẩn vector. Vì vậy không dùng riêng danh sách nearest-neighbor từ cell đó làm kết luận; các số co-occurrence ở trên lấy từ bảng cosine giữa các cặp từ.
