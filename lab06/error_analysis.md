# Error analysis — IMDb sentiment

Model: Self-attention classifier, max_length = 100. Test accuracy = 0,8200, macro F1 = 0,8198.

| Text (trích ngắn) | True | Predicted | Attention pattern | Phân tích |
|---|---|---|---|---|
| “pathetic ... performance, damn ... battle scenes are so bad ... story is so illogical” | negative | negative | pathetic (0,061), great (0,039), no (0,023) | Review có nhiều từ chê rõ ràng nên dự đoán đúng. Attention có cả great, nhưng phần còn lại của ngữ cảnh vẫn mang sắc thái tiêu cực. |
| “one among the very few Indian movies ... excellent skills ... great cinema” | positive | positive | artificially (0,038), excellent (0,035) | Review khen sức hấp dẫn và kỹ thuật làm phim. Attention có trọng số vào artificially trong câu chê pop culture, cho thấy token được chú ý không nhất thiết là từ khen trực tiếp. |
| “the use of facial image ... is superb ... truly rewarding ... highly recommend it” | positive | positive | throughout (0,047), director (0,034) | Các ý khen diễn xuất/đạo diễn và lời giới thiệu tích cực phù hợp với nhãn; model dự đoán đúng. |
| “no we don't, and either do you ... lacks tension and believability ... lackluster performance” | negative | positive | canadian (0,041), wild (0,032), lack (0,024) | Review phủ định lời khen mở đầu bằng giọng châm biếm rồi chê phim thiếu sức hút. Attention tập trung vào một số từ chủ đề/setting; có thể chưa đủ để model nắm thái độ phê bình xuyên suốt. |
| “Lucille Ball was one great comedienne ... wonderful ... until Mame ... that's about the nicest thing I can say” | negative | positive | great (0,047), perfect (0,037) | Review mở đầu bằng lời khen rồi chuyển sang chê phim bằng mỉa mai và phủ định. Attention cao ở great/perfect có thể góp phần vào dự đoán positive, nhưng đây chỉ là giả thuyết, không chứng minh nguyên nhân. |
| “Jerry ... sadistic ... makes Tom think his eyes are deceiving him ... the best part ...” | positive | negative | deceiving (0,046), think (0,033), milk (0,027) | Review có nhiều từ tiêu cực khi kể hành động trong cartoon, dù người viết đánh giá tích cực một số đoạn. Cảm xúc pha trộn khiến nhãn khó đoán từ từng token riêng lẻ. |

Confusion matrix có 1.028 negative bị dự đoán thành positive và 757 positive bị dự đoán thành negative. Lỗi false positive nhiều hơn. Phân phối test gần cân bằng nên class imbalance khó là nguyên nhân chính.

Các pattern trên cho biết model đặt weight ở đâu, nhưng không tự giải thích vì sao model dự đoán như vậy. Những lỗi này có thể liên quan tới châm biếm, phủ định và cảm xúc pha trộn; cần xem đó là giả thuyết đối chiếu với toàn bộ review, không kết luận chỉ từ heatmap.
