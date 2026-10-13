Ví dụ phân loại đúng

1. Text: “it is very entertaining to watch cassidy and falk as always ... the familiarness of columbo and his likability which scores high with viewers like myself.”
Nhãn thật: positive | Dự đoán: positive
Nhận xét: Người viết khen phim entertaining, nhân vật Columbo dễ xem và được yêu thích. Các từ/cụm từ tích cực thể hiện khá rõ cảm xúc chung nên mô hình dự đoán đúng.

2. Text: “i always loved this movie ... i still love it!!!! ... roy s. was a great actor ... the car chase is one of the all-time best.”
Nhãn thật: positive | Dự đoán: positive
Nhận xét: Review có nhiều dấu hiệu khen trực tiếp như loved, love, great và best. Dù có câu “not an oscar-worthy film”, phần đánh giá tổng thể vẫn tích cực; mô hình dự đoán đúng.

3. Text: “the appalling griswalds are just that - appalling. they are not funny ... every possible ‘comic’ situation is worked to death ... i feel cheated”
Nhãn thật: negative | Dự đoán: negative
Nhận xét: Người viết liên tục chê phim không hài, tình huống nhàm chán và cảm thấy bị lừa. Các từ ngữ tiêu cực nhất quán với nhãn negative nên mô hình dự đoán đúng.

Ví dụ phân loại sai

1. Text: “talking motion pictures had great difficulty ... this particular zane grey plot appears to be very weak ... the result is something that would not be acceptable by today's standards ... some of the actors performed well.”
Nhãn thật: negative | Dự đoán: positive
Nguyên nhân có thể: Các cụm “great difficulty” và “performed well” chứa từ thường mang sắc thái tích cực nhưng trong câu đầu “great” bổ nghĩa cho difficulty. Cùng với phủ định trong “not acceptable”, những ngữ cảnh này có thể khiến unigram tạo tín hiệu lẫn lộn. Mô hình có thể dựa vào một số từ nổi bật mà chưa nắm được phạm vi phủ định.

2. Text: “i have to say that they really do deserve respect ... they are truly good! ... i think this material isn't very good.”
Nhãn thật: negative | Dự đoán: positive
Nguyên nhân có thể: Review khen những người làm chương trình, nhưng chê nội dung/chất liệu của chương trình. Cảm xúc hướng tới nhiều đối tượng khác nhau trong cùng văn bản, nên từ tích cực như “respect” và “truly good” có thể lấn át ý đánh giá tiêu cực về nội dung. Đây là trường hợp cảm xúc pha trộn/khó xác định đối tượng được nhận xét.

3. Text: “the idea is to drive both trucks to a warehouse ... trouble is, one of the six ... is a holdout ... everything goes to hell ... the menacing, throbbing, blistering and totally sinister electronic soundtrack”
Nhãn thật: positive | Dự đoán: negative
Nguyên nhân có thể: Review tích cực và hào hứng về phim hành động, nhưng chứa nhiều từ thường gắn với cảm xúc tiêu cực như “trouble”, “hell”, “bad”, “violent” và “sinister”. Những từ này mô tả nội dung/phong cách phim chứ không nhất thiết thể hiện người viết không thích phim. Mô hình có thể dựa vào lexical cues mà chưa phân biệt được ngữ cảnh và thái độ tổng thể.

Nhận xét chung: Các lỗi cho thấy mô hình TF-IDF unigram + Linear SVM có thể bị ảnh hưởng bởi từ cảm xúc cục bộ, phủ định, cảm xúc trái chiều và sự khác nhau giữa cảm xúc của người viết với nội dung được mô tả. Phân tích nguyên nhân ở đây là giải thích khả dĩ từ văn bản; các ví dụ riêng lẻ không đủ để kết luận một nguyên nhân duy nhất.
