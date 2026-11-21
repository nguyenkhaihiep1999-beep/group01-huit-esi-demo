# Protocol thực nghiệm chuẩn (Đồng bộ OOF CV & Chống Overfitting)

Nguồn duy nhất: FedMML CSV giữ nguyên SHA-256 `94b159291b158728ef789e1ec2aef6b81f1fae19b2bd4970abdc2ab897835e0e`.

FedMML hoàn toàn tổng hợp, không có bệnh nhân thật. Các thuật toán dùng cùng dữ liệu cấu trúc (38 đầu vào $\rightarrow$ 46 đặc trưng sau tiền xử lý) và cùng 3 fold `StratifiedGroupKFold` theo `patient_id` mô phỏng. `patient_id`, `encounter_id`, `esi_level`, `chief_complaint`, `clinical_notes` và `arrival_timestamp` gốc bị loại khỏi ma trận đặc trưng.

## 1. Thứ tự thực nghiệm chuẩn

1. **Notebook 01 (`01_preprocessing.ipynb`)**: kiểm tra schema, EDA, làm sạch theo `BOUNDS`, giữ `NaN`, kiểm tra split cố định (`58 / 15 / 23` ID, overlap = 0), minh họa tiền xử lý 46 đặc trưng trên train và xuất CSV.
2. **Notebook 02 — Phần 1–8 (`run=fd81afda6ebcb7ea`)**:
   - Tìm siêu tham số chống overfitting (tối đa 12 cấu hình/thuật toán; LR có 8 cấu hình) trên 3 fold `StratifiedGroupKFold`.
   - Sinh xác suất Out-of-Fold (OOF) nghiêm ngặt trong 3 fold train cho từng thuật toán.
   - Tối ưu hệ số ngưỡng quyết định $\alpha \in [0.5, 2.0]^5$ và tỷ trọng trộn `SoftVoting` (`rf_weight` $\in \{0.25, 0.50, 0.75\}$) **hoàn toàn trên Train OOF CV** (`source="train_oof_cv"`). Tập Validation hoàn toàn không bị chạm vào lúc tối ưu.
   - Chấm điểm khách quan 1 lần trên Validation để chọn mô hình thắng cuộc (`Macro F1` $\rightarrow$ `Macro Recall` $\rightarrow$ `ROC-AUC macro OvR`).
   - Khóa SHA-256 (`model_lock.json`) trước khi mở Test đúng 1 lần và ghi biên nhận (`evaluation_receipt.json`).
3. **Notebook 02 — Phần 9–15 (`study=ceca64f75da026d8`)**: Thử 6 chiến lược cân bằng lớp trên cùng 3 fold train (giữ siêu tham số baseline, tối ưu ngưỡng bằng Train OOF CV). Chỉ đánh giá Test mới nếu vượt điều kiện Promotion trên Validation; nếu không, giữ nguyên kết quả baseline.
4. **Notebook 02 — Phần 16 (`study=2b141a247c81f58d`)**: Thử 4 bộ phân loại nhị phân chuyên biệt ranh giới ESI 4–5 (huấn luyện chỉ trên các dòng `{4, 5}` trong fold train, tối ưu ngưỡng OOF). Đo trực tiếp Binary ROC-AUC 4-vs-5 để kiểm chứng khả năng tách lớp của dữ liệu cấu trúc.
5. **Notebook 02 — Phần 17 (`study=ce0302bc4a643c2f`)**: Kiểm định độ ổn định qua 10 lần chia nhóm `GroupShuffleSplit(80/20)` hoàn toàn độc lập trên toàn bộ 96 `patient_id` (lồng 3-fold OOF CV trong mỗi split train, `patient_overlap = 0`, 0 tập holdout trùng lặp).

## 2. Đặc trưng & Tiền xử lý theo thuật toán

- `TrainQuantileClipper(0.5%, 99.5%)` fit riêng trên từng fold train.
- `log1p` cho 6 xét nghiệm lệch phải (`bnp`, `troponin`, `glucose`, `wbc`, `creatinine`, `lactate`).
- `LR` & `RandomForest`: điền `median` + `StandardScaler`. `XGBoost`: giữ nguyên `NaN` tự nhiên cho biến lâm sàng (Sparsity-aware split).
- 4 đặc trưng sinh lý/đếm (`shock_index`, `mean_arterial_pressure`, `pulse_pressure`, `danger_vital_count`), 2 biến đếm thiếu (`missing_vitals_count`, `missing_labs_count`), 3 cờ thiếu nhị phân không scale (`spo2_missing`, `troponin_missing`, `bnp_missing`) và 7 biến thời gian tuần hoàn (`sin/cos` + `is_weekend`).

## 3. Lịch sử test & Giới hạn

- Giữ split cố định `58 / 15 / 23` ID mô phỏng (`61.049 / 13.124 / 13.061` ca). Nhiều ID có tuổi/giới tính không nhất quán.
- Tập test đã từng được đánh giá trong lịch sử dự án (`historical_test_exposure = True`), không phải holdout mới chưa từng tiếp xúc.
- Xác suất chưa hiệu chỉnh (uncalibrated); kết quả không chứng minh tổng quát hóa lâm sàng tại bệnh viện thật.
