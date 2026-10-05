Dự đoán đúng 1
Observed: doctor - physician có cosine 0.7642, hạng 2 trong top-neighbors của doctor.
Expected: doctor và physician gần nhau.
Giải thích: Hai từ có nhiều ngữ cảnh dùng chung.
Evidence: context của doctor thường là the (191 lần), a (189), your (181); physician cũng thường đi với the (63), a (62), your (41).

Dự đoán đúng 2
Observed: hospital gần clinic với cosine 0.7009; các từ gần khác gồm hospice và ICU.
Expected: hospital gần các cơ sở/chăm sóc y tế.
Giải thích: Chúng xuất hiện trong ngữ cảnh y tế tương tự.
Evidence: top-neighbors của hospital là UPMC, Crittenton, hospice, clinic và ICU.

Dự đoán đúng 3
Observed: disease gần infection với cosine 0.8150.
Expected: hai từ liên quan về bệnh tật.
Giải thích: Chúng có khả năng cùng xuất hiện trong văn bản y khoa.
Evidence: disease còn gần autoimmune, chronic và osteoporosis; các context nổi bật gồm heart (101 lần), celiac (68), the (163).

Dự đoán bất ngờ 1
Observed: dentist đứng trên physician trong top-neighbors của doctor: 0.8118 so với 0.7642.
Expected: physician có thể đứng đầu vì gần nghĩa trực tiếp với doctor.
Giải thích: Dentist vẫn thuộc lĩnh vực y tế; thứ tự có thể do tần suất và cách dùng trong corpus web.
Evidence: doctor có nhiều context phổ biến hơn physician trong thống kê corpus.

Dự đoán bất ngờ 2
Observed: football gần replica (0.7888), shirts (0.7343) và shirt (0.7293), ngoài soccer (0.7614).
Expected: các từ về môn thể thao hoặc đội bóng sẽ đứng gần.
Giải thích: Corpus web có thể chứa nhiều trang sản phẩm/áo đấu liên quan đến football.
Evidence: chính top-neighbors có các từ replica, shirts, shirt; đây là dấu hiệu domain bias.

Dự đoán bất ngờ 3
Observed: banana gần tortilla (0.7656), slaw (0.7573), chutney (0.7570) và gravy (0.7474).
Expected: ban đầu chờ các từ tên trái cây.
Giải thích: Embedding phản ánh các context trong corpus; banana thường xuất hiện trong công thức hoặc mô tả món ăn.
Evidence: nhóm từ gần banana đều nghiêng về món ăn/nguyên liệu, không phải các loại trái cây.
