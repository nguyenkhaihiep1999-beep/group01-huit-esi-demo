# Thử nghiệm trọng số lớp và lấy mẫu: FedMML

Study ID: `ceca64f75da026d8`; baseline run: `fd81afda6ebcb7ea`.

## Kết luận

Phương án cuối chọn bằng validation: **XGBoost / baseline**.
Không phương án mới nào vượt điều kiện validation; giữ baseline và tái sử dụng kết quả test cũ, không suy luận test mới.

Test Macro F1: **0.77114 → 0.77114** (+0.000 điểm phần trăm).
Macro Recall: 0.76731; ROC-AUC macro OvR: 0.96024; Accuracy: 0.85790.
Không đổi lựa chọn mô hình sau khi xem kết quả test, kể cả khi kết quả test giảm.

## Thiết kế có kiểm soát

- Giữ nguyên bộ đặc trưng, preprocessing, split theo ID mô phỏng, seed 42 và siêu tham số của từng baseline.
- Sáu phương án × ba thuật toán × ba fold = 54 lượt fit CV; chỉ refit phương án CV tốt nhất của mỗi thuật toán.
- Preprocessing học trên các dòng train gốc của từng fold trước khi lấy mẫu; không lấy mẫu validation/test.
- Trọng số được tính lại trong từng fold train, chuẩn hóa trung bình bằng 1.
- Hai phương án lấy mẫu không cộng thêm class_weight/sample_weight để tránh cân bằng hai lần.
- Hệ số ngưỡng quyết định học hoàn toàn trên Out-of-Fold (OOF) CV của tập train; validation không bị chạm vào lúc tối ưu và chỉ quyết định có thay baseline cuối hay không.
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
| LogisticRegression | baseline | 0.74103 | 0.00328 | 0.00000 | 0.00000 |
| LogisticRegression | balanced_global | 0.70101 | 0.00534 | 0.25470 | 0.49181 |
| LogisticRegression | balanced_45_soft | 0.69405 | 0.00658 | 0.00168 | 0.00085 |
| LogisticRegression | balanced_5_boost | 0.59712 | 0.00417 | 0.26126 | 0.98256 |
| LogisticRegression | oversample_5_to_4 | 0.74275 | 0.00495 | 0.26588 | 0.47581 |
| LogisticRegression | undersample_4_to_5 | 0.73602 | 0.00730 | 0.26565 | 0.45189 |
| RandomForest | baseline | 0.76828 | 0.00005 | 0.18600 | 0.20490 |
| RandomForest | balanced_global | 0.76060 | 0.00312 | 0.22534 | 0.33957 |
| RandomForest | balanced_45_soft | 0.74797 | 0.00036 | 0.00058 | 0.00029 |
| RandomForest | balanced_5_boost | 0.62941 | 0.00117 | 0.26091 | 0.98493 |
| RandomForest | oversample_5_to_4 | 0.75901 | 0.00052 | 0.19009 | 0.20507 |
| RandomForest | undersample_4_to_5 | 0.72026 | 0.00383 | 0.24307 | 0.37852 |
| XGBoost | baseline | 0.75877 | 0.00228 | 0.22543 | 0.34183 |
| XGBoost | balanced_global | 0.75877 | 0.00228 | 0.22543 | 0.34183 |
| XGBoost | balanced_45_soft | 0.75060 | 0.00176 | 0.02614 | 0.01424 |
| XGBoost | balanced_5_boost | 0.66272 | 0.00214 | 0.26200 | 0.89636 |
| XGBoost | oversample_5_to_4 | 0.76486 | 0.00072 | 0.23394 | 0.32920 |
| XGBoost | undersample_4_to_5 | 0.74354 | 0.00326 | 0.26212 | 0.45265 |

## Validation: baseline và phương án chọn bằng CV

| Thuật toán | Phương án | Macro F1 | F1 ESI 5 | Precision ESI 5 | Recall ESI 5 |
|---|---|---:|---:|---:|---:|
| LogisticRegression | baseline | 0.74068 | 0.00000 | 0.00000 | 0.00000 |
| LogisticRegression | oversample_5_to_4 | 0.76008 | 0.15165 | 0.17177 | 0.13575 |
| RandomForest | baseline | 0.76411 | 0.18570 | 0.16263 | 0.21640 |
| RandomForest | baseline | 0.76411 | 0.18570 | 0.16263 | 0.21640 |
| XGBoost | baseline | 0.76857 | 0.18776 | 0.16826 | 0.21237 |
| XGBoost | oversample_5_to_4 | 0.76601 | 0.17600 | 0.16232 | 0.19220 |

## Test: baseline cuối và phương án cuối

| ESI | F1 trước | F1 sau | Precision trước | Precision sau | Recall trước | Recall sau |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.99690 | 0.99690 | 1.00000 | 1.00000 | 0.99383 | 0.99383 |
| 2 | 0.97512 | 0.97512 | 0.99377 | 0.99377 | 0.95717 | 0.95717 |
| 3 | 0.95261 | 0.95261 | 0.92728 | 0.92728 | 0.97938 | 0.97938 |
| 4 | 0.75243 | 0.75243 | 0.80481 | 0.80481 | 0.70645 | 0.70645 |
| 5 | 0.17864 | 0.17864 | 0.16157 | 0.16157 | 0.19973 | 0.19973 |

## Giới hạn và tái lập

Đây là thử nghiệm có phạm vi cố định, không chứng minh mọi chiến lược cân bằng đều đã được tối ưu.
FedMML hoàn toàn tổng hợp, không có bệnh nhân thật; giữ nguyên split 58/15/23 ID mô phỏng. Tuổi và giới tính không nhất quán trong nhiều ID, nên không trùng ID chưa chứng minh tính độc lập giữa bệnh nhân. Test đã từng được đánh giá và đã góp phần định hướng vấn đề ESI 4–5 cần cải thiện, nên không phải xác nhận độc lập.
Giữ nguyên việc loại chief_complaint và clinical_notes; thử nghiệm này không thêm tín hiệu từ văn bản. Xác suất chưa được calibration.
Artifact: `outputs/balancing/ceca64f75da026d8`. Các fold lưu số lượng lớp trước/sau, chỉ số lấy mẫu giới hạn trong phần fit, và overlap validation bằng 0.
Study manifest, selection lock, SHA-256 và test receipt cho phép Run All tái sử dụng kết quả đã xác minh.
Kết quả, bảng và biểu đồ nằm dưới từng cell trong notebook 02; chỉ giữ hai notebook của dự án.
