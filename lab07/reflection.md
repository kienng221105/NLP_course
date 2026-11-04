# Reflection — LAB 07

1. Self-attention thuần không biết token nằm ở vị trí nào. Trong permutation experiment, pooled representations của dog bites man và man bites dog giống nhau khi bỏ PE (distance 0); thêm PE thì distance là 0,2738. Kết quả minh họa PE đưa thông tin vị trí vào representation.

2. Mỗi head dùng projection riêng để tạo Q/K/V và có thể học một pattern khác. Trong ví dụ attention matrix, cosine giữa các heads nằm khoảng 0,147–0,403, nên chúng không hoàn toàn giống nhau nhưng cũng chưa thể gán thành các chức năng như syntax/semantics.

3. Residual giữ đường truyền trực tiếp của input; LayerNorm chuẩn hóa feature để hỗ trợ training; FFN biến đổi representation phi tuyến tại mỗi vị trí. Shape tracing cho thấy block giữ output shape (B,T,d_model) = (8,100,128). Train loss và validation loss đều giảm trong ba epoch của lần chạy này.

4. Test F1 macro của Transformer 4 heads là 0,7459, nhỉnh hơn LSTM 0,7437; thời gian train là 134,2 giây so với 236,6 giây, còn số tham số gần bằng nhau (3,973M). Trong cấu hình này Transformer nhanh hơn và F1 cao hơn một chút; một lần chạy không chứng minh nó luôn tốt hơn.

5. Từ max_length 32 lên 256, tỷ lệ token giữ lại tăng 13,83% lên 77,02%, test F1 macro tăng 0,7003 lên 0,8491, còn thời gian train tăng 60,0 lên 601,1 giây. Attention matrix tăng từ 1.024 lên 65.536 phần tử, tức 64 lần khi T tăng 8 lần.

6. Transformer học từ đầu vẫn phụ thuộc dữ liệu gán nhãn và chi phí tính toán; self-attention toàn cục tốn O(T²). Pretrained Transformer/BERT tận dụng pretraining trên lượng văn bản lớn rồi fine-tune cho task, là bước tiếp theo của lab.
