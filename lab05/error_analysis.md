# Error analysis — IMDb

Model được chọn theo validation macro F1: TF-IDF unigram + Logistic Regression. Trên test, accuracy = 0,8949 và F1 macro = 0,8949. Model này không dùng max_length.

| Text (trích ngắn) | True | Predicted | Phân tích |
|---|---|---|---|
| “this is one of the funniest movies i've ever saw ... it was funny ... hilarity ensues” | positive | positive | Các từ “funniest”, “funny” và “hilarity” là dấu hiệu tích cực rõ; dự đoán phù hợp với thái độ chung của review. |
| “an inventive, suspenseful ... taut, surprising and well-acted ... this reviewer found it enthralling ... a good film” | positive | positive | Nhiều từ khen về sự hồi hộp, diễn xuất và chất lượng phim củng cố cùng một nhãn. |
| “this film is really really bad ... very disappointed ... delivered nothing ... dreadful ... should be avoided” | negative | negative | Các đánh giá tiêu cực lặp lại xuyên suốt review nên mô hình dự đoán đúng. |
| “if i didn't have kids, i never would have seen this film ... significant improvement ... definitely better than average ... could do a lot worse” | positive | negative | Review dùng nhiều từ phủ định/tiêu cực khi so sánh phim trẻ em, nhưng kết luận phim này tốt hơn lựa chọn khác. Phủ định và so sánh làm tín hiệu từ vựng lẫn lộn; unigram không biểu diễn rõ phạm vi của “didn't” và ý tổng thể. |
| “i could not take my eyes off this movie ... not being a fan of the genre ... so-bad-it's-good” | positive | negative | “could not take my eyes off” mang ý cuốn hút, nhưng review có giọng châm biếm và nhiều từ tiêu cực về thể loại/phim. Mô hình có thể dựa vào các từ đó mà bỏ lỡ thái độ thích thú pha mỉa mai. |
| “state of the art ... more entertaining than most ... a knock-out ... sickening ... annoying ... repetitive” | positive | negative | Review vừa khen âm nhạc/phim vừa phê bình một số phần. Các từ tiêu cực nổi bật có thể lấn át lời khen; đây là cảm xúc pha trộn, không phải một nhãn từ vựng đơn giản. |

Confusion matrix có 592 negative bị dự đoán thành positive và 450 positive bị dự đoán thành negative. Tập dữ liệu gần cân bằng, nên class imbalance không phải lời giải thích chính cho các lỗi này.

Các lỗi được chọn chủ yếu liên quan tới phủ định, so sánh, giọng mỉa mai và cảm xúc pha trộn. Vì model được phân tích là TF-IDF unigram, không có bằng chứng để quy lỗi cho truncation hoặc cho một sequence model; hợp lý hơn là xem đây là giới hạn của tín hiệu từ vựng cục bộ và việc bỏ qua thứ tự/ngữ cảnh. Các review là đánh giá chủ quan, nên không đủ căn cứ kết luận nhãn bị nhiễu.
