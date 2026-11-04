# Prediction trước experiment

Dataset: IMDb — sentiment classification.

## 1. Transformer xử lý token song song hơn RNN không?

Prediction: Có, Transformer xử lý các vị trí song song hơn RNN.

Reason: Q/K/V và attention được tính theo các phép toán ma trận trên cả sequence; RNN phải truyền state qua từng timestep.

Confidence: Khá cao.

## 2. Bỏ positional encoding thì Transformer có biết thứ tự token không?

Prediction: Không tự biết thứ tự nếu thiếu positional information.

Reason: Self-attention dùng nội dung Q/K/V; vị trí không được cung cấp nếu không cộng positional encoding.

Confidence: Cao.

## 3. Hai câu cùng token nhưng khác thứ tự có phân biệt được nếu không có positional information không?

Prediction: Không có positional information, hai câu có thể cho pooled representation giống nhau.

Reason: Hai câu là hoán vị của cùng token embeddings; self-attention thuần permutation-equivariant và pooling không thứ tự làm mất vị trí.

Confidence: Khá cao.

## 4. Hai attention heads có nhất thiết học cùng pattern không?

Prediction: Không nhất thiết; các heads có thể học pattern khác nhau hoặc gần giống nhau.

Reason: Mỗi head có projection riêng, nhưng training không đảm bảo chúng học chức năng tách biệt.

Confidence: Vừa phải.

## 5. Tăng số heads có chắc chắn làm accuracy tăng không?

Prediction: Không chắc tăng.

Reason: Khi d_model cố định, tăng heads làm d_k giảm; kết quả còn phụ thuộc dữ liệu và quá trình train.

Confidence: Cao.

## 6. Transformer có luôn nhanh hơn RNN trên mọi sequence length không?

Prediction: Không phải lúc nào Transformer cũng nhanh hơn RNN.

Reason: Transformer song song hóa tốt hơn, nhưng full self-attention có chi phí O(T²) và overhead; với sequence ngắn RNN có thể nhanh hơn.

Confidence: Cao.

## Prediction verification sau khi chạy

| # | Prediction ban đầu | Kết quả quan sát | Đúng/Sai | Giải thích |
|---:|---|---|---|---|
| 1 | Transformer tính trên nhiều vị trí song song hơn RNN | Transformer train 134,2 giây, LSTM 236,6 giây tại max_length 64 | Phù hợp với lần chạy này | Kiến trúc Transformer cho phép các vị trí tương tác bằng phép toán ma trận; thời gian thực tế còn phụ thuộc máy và cấu hình. |
| 2 | Không có positional encoding thì thiếu thông tin thứ tự | Pooled distance của hai thứ tự là 0 khi không PE; 0,2738 khi có PE | Đúng trong experiment | Kết quả toy experiment phù hợp tính permutation-equivariant của self-attention thuần. |
| 3 | Cùng token khác thứ tự khó phân biệt nếu thiếu vị trí | Thí nghiệm dog bites man/man bites dog cho distance 0 không PE và khác biệt khi thêm PE | Đúng trong experiment | Positional information giúp representation phản ánh vị trí token. |
| 4 | Các heads có thể học pattern khác nhau nhưng không bắt buộc | Cosine giữa attention maps của các cặp heads khoảng 0,147–0,403 | Phù hợp một phần | Các pattern khác nhau trong mẫu đã xem; chưa đủ để gán cho head một vai trò ngôn ngữ. |
| 5 | Tăng heads không đảm bảo accuracy tăng | Validation F1 macro: 1 head 0,7465; 2 heads 0,7477; 4 heads 0,7485; 8 heads 0,7474 | Đúng trong experiment | F1 không tăng đều; 4 heads cao nhất trong lần chạy này. |
| 6 | Transformer không luôn nhanh hơn RNN ở mọi độ dài | Transformer train nhanh hơn tại base length 64, nhưng chi phí attention tăng theo T²; sweep length cho thấy T=256 train 601,1 giây | Chưa thể kết luận cho mọi trường hợp | Experiment chỉ đo LSTM/Transformer tại base length và Transformer ở các length; không đo LSTM trên toàn bộ length sweep. |
