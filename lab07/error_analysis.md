# Error analysis — IMDb sentiment

Model: Transformer mean pooling, d_model=128, 4 heads, max_length=64. Test accuracy=0,7469; macro F1=0,7459.

| Text (trích ngắn) | True | Predicted | Attention pattern | Phân tích |
|---|---|---|---|---|
| “one of the most refined and grand directors ... a true gift ... a wonder of refinement” | positive | positive | dvd (0,056), big (0,049), grand (0,048) | Review khen đạo diễn/phim xuyên suốt; các dấu hiệu tích cực xuất hiện sớm nên model dự đoán đúng. |
| “very literate, intelligent drama ... magnificent ... one of the great transition films ... do not miss it” | positive | positive | trying (0,048), literate (0,047), robards | Lời khen về diễn xuất và chất lượng phim khá nhất quán, phù hợp nhãn positive. |
| “starts out like gangbusters ... unfortunately ... plod its weary way ... standard chan formula” | negative | negative | moreland (0,067), unfortunately (0,065) | Review có khen Mantan Moreland, nhưng đánh giá chính là phim chậm và theo công thức; model vẫn phân loại đúng. |
| “cute and blonde ... too sweet ... material lets her down” | positive | negative | cute (0,056), torch (0,055), world (0,043) | Review phê bình diễn viên và kịch bản ở phần đầu; lời khen quan trọng ở cuối là “I do like this movie for its empowerment of women”, nằm sau 64-token limit. Truncation là một nguyên nhân có thể giải thích lỗi này. |
| “it might be brilliant literature ... this cinema-adaption really sucks ... fighting against sleep” | negative | positive | wherein (0,070), books (0,068), brilliant | Review chê phim rõ, nhưng một số token nói về sách/văn học và brilliant có thể tạo tín hiệu positive cục bộ. Từ attention pattern không thể khẳng định đây là nguyên nhân duy nhất. |
| “very poor quality compilation ... now having seen a beautiful widescreen version ... deserves to be seen ... high praise” | positive | negative | poor (0,081), three (0,062), best (0,051) | Review khen phim tổng thể nhưng bắt đầu bằng cụm “poor quality” nói về bản phim cũ. Phần kết luận tích cực nằm sau 64 tokens nên có thể bị cắt khỏi input. |

Confusion matrix có 924 negative bị dự đoán thành positive và 1.586 positive bị dự đoán thành negative. Test split gần cân bằng, nên class imbalance khó là nguyên nhân chính.

Các lỗi gợi ý vai trò của truncation và tín hiệu từ vựng cục bộ. Attention pattern là trung bình trọng số theo heads/query, chỉ cho biết model phân phối attention ở đâu; nó không chứng minh các token đó là lý do model đưa ra nhãn.
