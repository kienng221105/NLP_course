# Error analysis — IMDb sentiment

**Model:** `google/bert_uncased_L-2-H-128_A-2`, fine-tuned 3 epoch trên 25% train (7.436 review), max_length 128, batch 32. Test sample phân tầng có 2.000 review (1.000 mỗi nhãn). Accuracy = 0,783; macro F1 = 0,7825.

Confusion matrix (hàng là nhãn thật, cột là nhãn dự đoán):

|  | Dự đoán negative | Dự đoán positive |
|---|---:|---:|
| Thật negative | 833 | 163 |
| Thật positive | 271 | 733 |

### Các lỗi được chọn

| Nhóm lỗi | True | Predicted | Confidence | Trích nội dung | Loại lỗi và giả thuyết |
|---|---|---|---:|---|---|
| False positive | negative | positive | 0,8848 | “I hadn't heard of this film… finest Christmas specials… highest regard… but it seems the man has his faults… the film… collapses under the weight of its ambitions.” | **Sentiment lẫn lộn / phụ thuộc ngữ cảnh.** Mở đầu có nhiều từ khen về họa sĩ và hoạt hình; đánh giá chung sau đó chê phim. Model có thể dựa quá nhiều vào tín hiệu khen xuất hiện sớm. |
| False positive | negative | positive | 0,8829 | “Just watched on UbuWeb… fascinating to watch… That's not to say this was any good… Worth a look.” | **Phủ định và đánh giá mâu thuẫn.** “Fascinating” và “worth a look” là tín hiệu tích cực, nhưng “not any good” mới thể hiện đánh giá trực tiếp. Model có thể không nắm đúng phạm vi phủ định. |
| False positive | negative | positive | 0,8791 | “Stan Laurel and Oliver Hardy… very interesting… but… very dire… direly unfunny.” | **Phụ thuộc ngữ cảnh xa.** Phần đầu nêu thông tin và khen một vài phim khác; kết luận về phim đang review là tiêu cực. Các tín hiệu tích cực không thuộc cùng đối tượng đánh giá. |
| False negative | positive | negative | 0,9029 | “black tar can't be snorted… abuse… not bad… quite heavy… no big deal.” | **Từ chuyên biệt và phủ định.** Nội dung nói về ma túy/lạm dụng có nhiều từ mang sắc thái nặng; các cụm “not bad” và “no big deal” làm sắc thái review tích cực/nhẹ hơn. Model có thể đọc chủ đề thành sentiment. |
| False negative | positive | negative | 0,9010 | “A horror movie… killer… curse… people… killed… The plot is old hat but this isn't bad… the script doesn't insult your intelligence… not bad at all. I give it a 7.” | **Phủ định và tín hiệu trái chiều.** Review có nhiều từ tiêu cực mô tả cốt truyện, nhưng kết luận khen vừa phải và chấm 7. Model có thể ưu tiên từ khóa về horror/kill/old hat hơn phần tổng kết. |
| False negative | positive | negative | 0,9005 | “maybe i need to have my head examined, but i thought this was a pretty good movie… not too bad… action… really good… all in all… about a 7/10.” | **Phủ định và sentiment lẫn lộn.** Câu có nhiều hedge như “not too bad”, “seen worse” cùng vài điểm chê, dù đánh giá cuối tích cực. Model có thể diễn giải các cụm phủ định như chê mạnh. |

Hai lỗi sai có confidence cao nhất là hai false negative đầu bảng, lần lượt 0,9029 và 0,9010. Nhóm confidence >=0,90 có 103 dự đoán, accuracy 0,9515; vẫn có 5 lỗi trong nhóm này.

### Confidence theo nhóm

| Nhóm | Số dự đoán | Accuracy | Confidence trung bình |
|---|---:|---:|---:|
| Thấp (<0,70) | 430 | 0,5698 | 0,5996 |
| Trung bình (0,70–0,90) | 1.467 | 0,8337 | 0,8344 |
| Cao (>=0,90) | 103 | 0,9515 | 0,9046 |

Confidence cao thường đúng hơn trong sample này nhưng không bảo đảm đúng; các false negative trên là phản ví dụ cụ thể. Các giải thích trong bảng là giả thuyết dựa trên nội dung review và đầu ra model, không khẳng định đây là cơ chế nội bộ duy nhất.

### Câu hỏi bắt buộc

**Q1. Model nào hưởng lợi nhiều hơn khi training data ít?** Ở 10%, pretrained đạt validation macro F1 0,5995; from scratch đạt 0,3342 trên cùng 2.975 review. Đây là bằng chứng trong cấu hình này rằng pretrained model hữu ích hơn khi nhãn ít.

**Q2. Pretrained tốt hơn from scratch chứng minh điều gì?** Kết quả ủng hộ việc biểu diễn đã pretrain có thể chuyển sang IMDb sentiment và giúp model học task với dữ liệu gán nhãn ít hơn. Nó không chứng minh model hiểu ngôn ngữ như con người hay luôn tốt hơn ở domain khác.

**Q3. Frozen gần bằng fine-tuned có ý nghĩa gì?** Khi đó feature pretrained có thể đã đủ cho task, nên cập nhật encoder đem lại ít cải thiện. Trong kết quả này frozen macro F1 0,5559 thấp hơn fine-tuning 0,6950 nên chênh lệch còn đáng kể.

**Q4. Fine-tuning tốt hơn frozen nhiều gợi ý gì?** Encoder cần thích nghi với tín hiệu sentiment và cách dùng từ trong IMDb; classification head một mình chưa khai thác hết biểu diễn có sẵn.

**Q5. Vì sao performance có thể giảm ở domain khác?** Từ vựng, văn phong và cách biểu hiện sentiment thay đổi theo domain. Ở sample hiện tại, macro F1 giảm nhẹ từ 0,7825 trên IMDb xuống 0,7733 trên Amazon, không phải mức giảm mạnh; sample Amazon chỉ có 1.000 review nên không khái quát được mọi product review.

**Q6. Có nên tiếp tục tăng sequence length?** Từ 128 lên 256, macro F1 tăng 0,0296 nhưng thời gian train tăng từ 95,2 lên 214,0 giây; số điểm attention mỗi layer tăng 4 lần. Chỉ nên tăng tiếp nếu phần context dài hơn đem lại lợi ích đủ cho task và chi phí thêm.
