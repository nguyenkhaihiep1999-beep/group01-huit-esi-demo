# Protocol thực nghiệm mới

Nguồn duy nhất: FedMML CSV giữ nguyên SHA-256 `94b159291b158728ef789e1ec2aef6b81f1fae19b2bd4970abdc2ab897835e0e`.

Ba thuật toán dùng cùng dữ liệu có cấu trúc và cùng fold bệnh nhân. Patient ID, encounter ID, target và văn bản tự do bị loại khỏi ma trận đặc trưng.

## Thứ tự

1. Notebook 01: kiểm tra schema, EDA, làm sạch, null/outlier, minh họa chuẩn hóa/mã hóa trên train, kiểm tra split và xuất dữ liệu.
2. Notebook 02: 6 cấu hình mỗi thuật toán trên cùng 3 fold StratifiedGroupKFold, refit cấu hình tốt nhất trên train.
3. Chọn thuật toán theo Macro F1 validation; tie-break Macro Recall, ROC-AUC macro OvR, tên thuật toán ổn định.
4. Xác minh SHA-256 các pipeline và CSV, sau đó đánh giá cả ba mô hình đã khóa trên test.
5. Ghi receipt cùng hash lock, model, report và prediction. Run All lại cùng code/config/data dùng kết quả có kiểm chứng.

## Metric

- Macro F1: trung bình F1 của năm lớp, mỗi lớp cùng trọng số.
- Macro Recall: trung bình Recall của năm lớp; khi đủ support năm lớp, bằng balanced accuracy.
- ROC-AUC macro OvR: AUC mỗi lớp so với bốn lớp còn lại, sau đó lấy trung bình không trọng số. Cột xác suất đúng thứ tự ESI 1–5.
- Confusion matrix: hàng là ESI thực tế, cột là ESI dự đoán, ô chứa số lượt khám.
- Accuracy và Recall ESI 1/2 bổ sung để đọc mức đánh đổi của từng thuật toán.

## Các quyết định cố định

Seed 42; ba fold; sáu cấu hình mỗi thuật toán; outlier quantile 0,5%/99,5%; 26 đặc trưng trước mã hóa. Không dùng early stopping với test và không chọn mô hình theo test.

Quy tắc miền giá trị là quy tắc kỹ thuật kế thừa từ xử lý FedMML, không phải ngưỡng lâm sàng đã chứng nhận. Giá trị bất hợp lệ chuyển thành null; clipping, median, scaler và encoder đều học riêng trên train của từng fold.

## Lịch sử test

Giữ split patient_id cũ: 58/15/23 bệnh nhân và 61.049/13.124/13.061 lượt khám. Test đã từng được đánh giá; xóa file cũ không làm dữ liệu trở thành holdout mới. Kết luận tổng quát hóa mạnh hơn cần bệnh nhân hoặc dữ liệu ngoài chưa được sử dụng.

## Thử trọng số/lấy mẫu bổ sung

Notebook 02 phần 9–15 chạy thử nghiệm có kiểm soát: giữ nguyên các đặc trưng và siêu tham số từng baseline, khảo sát sáu phương án cân bằng trên đúng ba fold train đã dùng. Không thêm chief_complaint/clinical_notes, không đổi split và không mở rộng tìm kiếm siêu tham số.

- baseline: giữ nguyên class_weight/weight_mode đã chọn trước.
- balanced_global: trọng số lớp n/(5×số mẫu lớp), tính riêng mỗi fold train, chuẩn hóa trung bình sample_weight bằng 1.
- balanced_45_soft: nhân trọng số balanced ESI 4 với 0,85 và ESI 5 với 0,5, chuẩn hóa lại.
- balanced_5_boost: nhân trọng số balanced ESI 5 với 2, chuẩn hóa lại.
- oversample_5_to_4: lặp ngẫu nhiên mẫu ESI 5 đến số lượng ESI 4; các lớp khác giữ nguyên.
- undersample_4_to_5: giữ ngẫu nhiên ESI 4 bằng số lượng ESI 5; các lớp khác giữ nguyên.

Trọng số hoặc lấy mẫu chỉ áp dụng trong fit của classifier, sau preprocessing đã fit trên các dòng train gốc. Không cộng thêm class_weight/sample_weight trong hai phương án lấy mẫu. Ghi số mẫu từng lớp trước/sau, hash chỉ số lấy mẫu và overlap với validation cho từng fold.

Chọn phương án mỗi thuật toán bằng Macro F1 CV, tie-break Macro Recall và thứ tự phương án cố định. Refit chỉ phương án CV tốt nhất rồi đánh giá validation. Điều kiện thay baseline cuối được khai báo trước chạy: Macro F1 validation tăng, F1 ESI 5 không giảm, Recall ESI 1/2 giảm tối đa 1 điểm phần trăm. Chọn thuật toán theo Macro F1 validation trong các phương án đạt điều kiện.

Khóa lựa chọn và SHA-256 trước suy luận test mới. Chỉ đánh giá phương án cuối được thay vào một lần; nếu không đạt điều kiện thì tái sử dụng baseline test cũ. Kết quả test không dùng để đảo lựa chọn. Vì điểm yếu ESI 4–5 đã được thấy trên test trước lần thử này, kết quả là benchmark phát triển có test đã tiếp xúc, không phải xác nhận độc lập mới.
