# Error Analysis — CoNLL-2003 NER

**Model:** `google/bert_uncased_L-2_H-128_A-2`, fine-tuned 5 epochs, max_length 128, batch 32, first-subword labels with `-100` for special/continuation pieces. Test entity-level F1 = `0,6463`; token accuracy = `0,9366`.

### False positives

| Text / span | Gold | Predicted | Error type | Nhận xét dựa trên câu |
|---|---|---|---|---|
| Headline “SOCCER - JAPAN GET LUCKY WIN, CHINA IN SURPRISE DEFEAT.” — `CHINA` | B-PER | B-LOC | Wrong entity type | Trong headline thể thao, `CHINA` chỉ đội tuyển quốc gia; model chọn LOC theo nghĩa địa lý. Đây là trường hợp mơ hồ theo domain/context. |
| `AL-AIN, United Arab Emirates` — `United` | Một span LOC: `United Arab Emirates` | Một span LOC: `United` | Boundary / spurious partial span | Model mở entity ở `United` nhưng kết thúc quá sớm. |
| Cùng câu — `Arab` | Thuộc span LOC bắt đầu ở `United` | B-LOC riêng | Boundary | Continuation token bị gán lại thành entity mới. |
| Cùng câu — `Emirates` | Thuộc span LOC `United Arab Emirates` | B-LOC riêng | Boundary | Cùng lỗi chia một địa danh nhiều từ thành các span một từ. |
| “Japan began the defence of their Asian Cup title…” — `Asian` | Một span MISC: `Asian Cup` | B-MISC riêng cho `Asian` | Boundary | Model nhận được loại entity nhưng bỏ `Cup` khỏi span hoàn chỉnh. |

### False negatives

| Text / span | Gold | Predicted | Error type | Nhận xét dựa trên câu |
|---|---|---|---|---|
| Headline “... CHINA IN SURPRISE DEFEAT.” — `CHINA` | B-PER | B-LOC | Wrong entity type / missed gold | Đây là cùng bề mặt từ với FP phía trên; context thể thao dùng quốc gia để chỉ đội tuyển. |
| `United Arab Emirates` | Một span LOC ba token | Ba span LOC một token | Missing exact entity | Các token riêng lẻ có nhãn LOC nhưng không tạo thành gold span chính xác. |
| `Asian Cup` trong câu về Japan thắng Syria | Một span MISC hai token | Chỉ `Asian` được ghép thành span | Missing exact entity / boundary | Entity-level đánh dấu lỗi dù một phần token đã đúng. |
| `Uzbek` trong câu “... Uzbek striker Igor Shkvyrin ...” | B-MISC | O | Missing entity / rare demonym | Demonym ít phổ biến có thể khó hơn tên người và địa điểm quen thuộc. |
| `Asian Games` trong câu “Despite winning the Asian Games title...” | Một span MISC hai token | Span không khớp gold | Missing entity / boundary | Cụm giải đấu có nhiều từ; model không khôi phục đúng toàn span. |

### Boundary errors

| Context | Gold span | Predicted span | Sai lệch |
|---|---|---|---|
| `United Arab Emirates` | `United Arab Emirates` — LOC | `United` — LOC | Span kết thúc quá sớm |
| `United Arab Emirates` | `United Arab Emirates` — LOC | `Arab` — LOC | Token giữa bị biến thành span riêng |
| `United Arab Emirates` | `United Arab Emirates` — LOC | `Emirates` — LOC | Token cuối bị biến thành span riêng |
| `Asian Cup` trong câu về Japan | `Asian Cup` — MISC | `Asian` — MISC | Bỏ token cuối |
| `Asian Cup` trong cùng câu | `Asian Cup` — MISC | `Cup` — ORG | Boundary và entity type cùng sai |

### Context analysis: cùng surface form, khác nhãn gold

| Surface form | Context 1 | Type 1 | Context 2 | Type 2 |
|---|---|---|---|---|
| Arizona | “... entertain Arizona.” (đội trong lịch thi đấu) | ORG | “DALLAS AT ARIZONA” (địa điểm trong lịch) | LOC |
| Atlanta | “ATLANTA 10 8 .556 7” (bảng kết quả đội) | ORG | “... Court of Appeals in Atlanta ...” | LOC |
| American | “AMERICAN FOOTBALL CONFERENCE” | MISC | “... British Airways and American drop 168 slots ...” | ORG |

### Interpretation

Baseline đạt entity F1 cao hơn Transformer ở test (`0,7334` so với `0,6463`). Baseline dùng capitalization, shape và cửa sổ context ±2; Transformer dùng một checkpoint nhỏ, uncased, nên mất tín hiệu chữ hoa quan trọng trong NER. Transformer có token accuracy cao (`0,9366`) nhưng entity F1 thấp hơn, cho thấy nhiều token đúng chưa đủ để khớp span hoàn chỉnh. Các ví dụ trên gợi ý sai boundary, entity type mơ hồ trong tin thể thao và demonym/entity type MISC là các điểm cần xem lại.
