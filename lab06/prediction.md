# Prediction trước experiment

Dataset: IMDb.

## 1. Key có dot product lớn nhất sẽ nhận attention weight thế nào?

Prediction: Key đó sẽ nhận weight lớn nhất.

Reason: Softmax giữ nguyên thứ tự của scores; score cao hơn thì xác suất sau softmax cũng cao hơn.

Confidence: Cao.

## 2. Nếu tất cả scores bằng nhau, weights sẽ ra sao?

Prediction: Các key nhận weight bằng nhau; với bốn scores thì mỗi weight là 0,25.

Reason: Các exponential trong softmax bằng nhau và được chuẩn hóa cùng một tổng.

Confidence: Cao.

## 3. Nếu một score lớn hơn hẳn các score còn lại thì attention thay đổi thế nào?

Prediction: Distribution sẽ tập trung chủ yếu vào key có score lớn nhất, nhưng các key khác vẫn có thể nhận weight nhỏ khác 0.

Reason: Exponential làm chênh lệch score thể hiện rõ hơn trong softmax.

Confidence: Cao.

## 4. Nếu sequence tăng từ 10 lên 1.000 tokens, computational cost thay đổi thế nào?

Prediction: Attention matrix tăng từ 100 lên 1.000.000 phần tử, tức gấp 10.000 lần.

Reason: Full self-attention lưu/tính tương tác giữa mọi cặp token, nên số phần tử tăng theo T².

Confidence: Cao.

## 5. Attention có luôn chọn đúng token mà con người xem là quan trọng nhất không?

Prediction: Không nhất thiết.

Reason: Weights phản ánh độ tương thích giữa query và key của model; chúng không đảm bảo trùng với cách con người giải thích câu.

Confidence: Cao.

## Đối chiếu sau khi chạy

Các kết quả minh họa phù hợp với cơ chế: Query A đặt weight cao nhất cho movie (0,373), còn Query B cho story (0,373). Với scores [1, 2, 3], temperature giảm từ 2 xuống 0,5 làm max weight tăng từ 0,506 lên 0,867. Attention classifier đạt test F1 macro 0,8198, cao hơn LSTM 0,7936 trong lần chạy này.

Heatmap không phải lúc nào cũng trùng với cách người đọc xác định ý chính. Chẳng hạn model dự đoán sai câu “I like this movie” thành negative dù attention tập trung nhiều vào like; điều đó cho thấy weight không tự nó giải thích được prediction.
