# Thử nghiệm trọng số lớp và lấy mẫu: FedMML

Study ID: `59268713ed51c68c`; baseline run: `71eafec26795d19a`.

## Kết luận

Phương án cuối chọn bằng validation: **XGBoost / oversample_5_to_4**.
Phương án mới vượt điều kiện validation và được đánh giá test một lần.

Test Macro F1: **0.77198 → 0.77083** (-0.115 điểm phần trăm).
Macro Recall: 0.77176; ROC-AUC macro OvR: 0.96087; Accuracy: 0.84320.
Không đổi lựa chọn mô hình sau khi xem kết quả test, kể cả khi kết quả test giảm.

## Thiết kế có kiểm soát

- Giữ nguyên 26 đặc trưng, preprocessing, split bệnh nhân, seed 42 và siêu tham số của từng baseline.
- Sáu phương án × ba thuật toán × ba fold = 54 lượt fit CV; chỉ refit phương án CV tốt nhất của mỗi thuật toán.
- Preprocessing học trên các dòng train gốc của từng fold trước khi lấy mẫu; không lấy mẫu validation/test.
- Trọng số được tính lại trong từng fold train, chuẩn hóa trung bình bằng 1.
- Hai phương án lấy mẫu không cộng thêm class_weight/sample_weight để tránh cân bằng hai lần.
- Chọn phương án bằng Macro F1 CV; validation dùng để quyết định có thay baseline cuối hay không.
- Điều kiện thay baseline: Macro F1 validation tăng, F1 ESI 5 không giảm, Recall ESI 1/2 giảm tối đa 1 điểm phần trăm.

## Sáu phương án

1. baseline: giữ nguyên cách cân bằng đã chọn trong benchmark trước.
2. balanced_global: trọng số nghịch đảo tần suất lớp trên toàn phần train của fold.
3. balanced_45_soft: balanced_global, nhân trọng số ESI 4 với 0,85 và ESI 5 với 0,5, rồi chuẩn hóa lại.
4. balanced_5_boost: balanced_global, nhân trọng số ESI 5 với 2, rồi chuẩn hóa lại.
5. oversample_5_to_4: lặp ngẫu nhiên ESI 5 đến số lượng ESI 4; các lớp khác giữ nguyên.
6. undersample_4_to_5: giữ ngẫu nhiên ESI 4 bằng số lượng ESI 5; các lớp khác giữ nguyên.

## Cross-validation trên train

| Thuật toán | Phương án | Macro F1 CV | SD | F1 ESI 5 CV | Recall ESI 5 CV |
|---|---|---:|---:|---:|---:|
| LogisticRegression | baseline | 0.74135 | 0.00377 | 0.00000 | 0.00000 |
| LogisticRegression | balanced_global | 0.69497 | 0.00751 | 0.25299 | 0.48604 |
| LogisticRegression | balanced_45_soft | 0.68997 | 0.00816 | 0.00168 | 0.00085 |
| LogisticRegression | balanced_5_boost | 0.58839 | 0.00179 | 0.25974 | 0.97878 |
| LogisticRegression | oversample_5_to_4 | 0.74446 | 0.00404 | 0.26415 | 0.46800 |
| LogisticRegression | undersample_4_to_5 | 0.73681 | 0.00626 | 0.26309 | 0.44531 |
| RandomForest | baseline | 0.76471 | 0.00065 | 0.17255 | 0.18004 |
| RandomForest | balanced_global | 0.75288 | 0.00314 | 0.21216 | 0.30313 |
| RandomForest | balanced_45_soft | 0.73974 | 0.00398 | 0.00000 | 0.00000 |
| RandomForest | balanced_5_boost | 0.62876 | 0.00080 | 0.26209 | 0.97709 |
| RandomForest | oversample_5_to_4 | 0.76198 | 0.00205 | 0.18230 | 0.19156 |
| RandomForest | undersample_4_to_5 | 0.72500 | 0.00439 | 0.25145 | 0.39778 |
| XGBoost | baseline | 0.76310 | 0.00070 | 0.22096 | 0.31569 |
| XGBoost | balanced_global | 0.76310 | 0.00070 | 0.22096 | 0.31569 |
| XGBoost | balanced_45_soft | 0.75400 | 0.00276 | 0.03059 | 0.01687 |
| XGBoost | balanced_5_boost | 0.67672 | 0.00277 | 0.25973 | 0.84702 |
| XGBoost | oversample_5_to_4 | 0.76763 | 0.00118 | 0.22223 | 0.29312 |
| XGBoost | undersample_4_to_5 | 0.74240 | 0.00287 | 0.25384 | 0.44183 |

## Validation: baseline và phương án chọn bằng CV

| Thuật toán | Phương án | Macro F1 | F1 ESI 5 | Precision ESI 5 | Recall ESI 5 |
|---|---|---:|---:|---:|---:|
| LogisticRegression | baseline | 0.73311 | 0.00000 | 0.00000 | 0.00000 |
| LogisticRegression | oversample_5_to_4 | 0.73166 | 0.24904 | 0.17438 | 0.43548 |
| RandomForest | baseline | 0.75970 | 0.17446 | 0.15149 | 0.20565 |
| RandomForest | baseline | 0.75970 | 0.17446 | 0.15149 | 0.20565 |
| XGBoost | baseline | 0.75333 | 0.20839 | 0.15360 | 0.32392 |
| XGBoost | oversample_5_to_4 | 0.76221 | 0.21837 | 0.17004 | 0.30511 |

## Test: baseline cuối và phương án cuối

| ESI | F1 trước | F1 sau | Precision trước | Precision sau | Recall trước | Recall sau |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.98462 | 0.98433 | 0.98160 | 1.00000 | 0.98765 | 0.96914 |
| 2 | 0.97335 | 0.97790 | 0.97433 | 0.99055 | 0.97238 | 0.96557 |
| 3 | 0.94503 | 0.95409 | 0.96089 | 0.92920 | 0.92968 | 0.98035 |
| 4 | 0.75043 | 0.70462 | 0.75565 | 0.81859 | 0.74529 | 0.61850 |
| 5 | 0.20646 | 0.23319 | 0.18026 | 0.18175 | 0.24157 | 0.32524 |

## Giới hạn và tái lập

Đây là thử nghiệm có phạm vi cố định, không chứng minh mọi chiến lược cân bằng đều đã được tối ưu.
FedMML có 96 bệnh nhân; giữ nguyên split 58/15/23 bệnh nhân. Test đã từng được đánh giá và đã góp phần định hướng vấn đề ESI 4–5 cần cải thiện, nên không phải xác nhận độc lập.
Giữ nguyên việc loại chief_complaint và clinical_notes; thử nghiệm này không thêm tín hiệu từ văn bản. Xác suất chưa được calibration.
Artifact: `outputs/balancing/59268713ed51c68c`. Các fold lưu số lượng lớp trước/sau, chỉ số lấy mẫu giới hạn trong phần fit, và overlap validation bằng 0.
Study manifest, selection lock, SHA-256 và test receipt cho phép Run All tái sử dụng kết quả đã xác minh.
Kết quả, bảng và biểu đồ nằm dưới từng cell trong notebook 02; chỉ giữ hai notebook của dự án.
