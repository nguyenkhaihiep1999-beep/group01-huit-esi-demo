# Kết quả thực nghiệm FedMML: ba thuật toán và soft voting RF/XGB

Run ID: fd81afda6ebcb7ea

Tập test: 13,061 ca tổng hợp, 23 ID mô phỏng. FedMML không có bệnh nhân thật.
Mô hình được chọn trên validation: **XGBoost**. Lựa chọn được khóa trước khi đánh giá test.

| Thuật toán | Macro F1 | Macro Recall | ROC-AUC macro OvR | Accuracy | Recall ESI 1 | Recall ESI 2 |
|---|---:|---:|---:|---:|---:|---:|
| LogisticRegression | 0.7466 | 0.7534 | 0.9618 | 0.9024 | 0.9074 | 0.9668 |
| RandomForest | 0.7700 | 0.7687 | 0.9608 | 0.8522 | 0.9938 | 0.9660 |
| XGBoost | 0.7711 | 0.7673 | 0.9602 | 0.8579 | 0.9938 | 0.9572 |
| SoftVoting | 0.7742 | 0.7707 | 0.9606 | 0.8666 | 0.9938 | 0.9608 |

## Quy trình

- Dùng duy nhất CSV FedMML hiện có; không sử dụng hoặc tuyên bố sử dụng MIMIC-IV-ED.
- Giữ cùng split theo patient_id mô phỏng của nghiên cứu trước để so sánh ổn định; không trùng ID giữa các tập. Điều này chưa chứng minh các bệnh nhân độc lập.
- Tối ưu tối đa 12 cấu hình mỗi thuật toán (siết điều chuẩn chống overfitting), 3 fold StratifiedGroupKFold theo ID; chọn theo Macro F1 CV.
- Clipping quantile 0.5%/99.5%, log1p cho 6 xét nghiệm lệch phải, scaler và one-hot fit trong từng fold train; LR/RF điền median kèm biến đếm thiếu, XGBoost giữ NaN tự nhiên.
- Ba cờ thiếu không chuẩn hóa; bổ sung 4 chỉ số sinh lý/đếm nguy hiểm và tọa độ thời gian tuần hoàn sin/cos (46 đặc trưng sau mã hóa).
- Toàn bộ trọng số quyết định (class multipliers) và tỷ trọng soft voting RF/XGB được chọn hoàn toàn trên Out-of-Fold (OOF) CV của tập train; validation hoàn toàn không bị chạm vào lúc tối ưu và chỉ dùng để chọn mô hình cuối.
- Patient ID, encounter ID, target và văn bản tự do bị loại khỏi đầu vào mô hình.
- SHA-256 của dữ liệu và mọi artifact được xác minh trước khi mở test; lưu receipt để không tự đánh giá lại.

## Giới hạn

FedMML này có 87.234 ca tổng hợp với 96 patient_id mô phỏng; tuổi và giới tính không nhất quán trong nhiều ID. Các ca không đại diện cho 87.234 mẫu bệnh nhân thật độc lập. Sinh hiệu và xét nghiệm được sinh theo nhãn ESI, nên benchmark đo mức học quy luật dữ liệu sinh.
Dữ liệu đã được khám phá và tập test đã được đánh giá ở nghiên cứu trước; đây là benchmark phát triển lại, không phải holdout mới chưa từng xem.
Xác suất chưa được calibration. Kết quả là nghiên cứu trên FedMML, chưa chứng minh khả năng tổng quát hóa trên bệnh nhân hoặc bệnh viện mới.

## Tệp kết quả

- Thư mục run: outputs/experiments/fd81afda6ebcb7ea
- model_lock.json; evaluation_receipt.json; test_metrics.json; test_comparison.csv.
- models/: ba pipeline, soft voting và SHA-256 trong manifest.
- figures/: so sánh metric, ma trận nhầm lẫn, ROC từng lớp.
- notebooks/01_preprocessing.ipynb và notebooks/02_model_comparison.ipynb có output dưới từng cell.

## Thử nghiệm cân bằng bổ sung

Phương án cuối chọn bằng validation: **XGBoost / baseline**. Test Macro F1 = **0.77114**; thay đổi **+0.000 điểm phần trăm** so với baseline cuối. Chi tiết và mọi phương án: [BALANCING_RESULTS.md](BALANCING_RESULTS.md).
