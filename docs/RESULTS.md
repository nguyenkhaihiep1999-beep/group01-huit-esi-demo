# Kết quả thực nghiệm FedMML: XGBoost, Logistic Regression, Random Forest

Run ID: 71eafec26795d19a

Tập test: 13,061 lượt khám, 23 bệnh nhân.
Mô hình được chọn trên validation: **RandomForest**. Lựa chọn được khóa trước khi đánh giá test.

| Thuật toán | Macro F1 | Macro Recall | ROC-AUC macro OvR | Accuracy | Recall ESI 1 | Recall ESI 2 |
|---|---:|---:|---:|---:|---:|---:|
| LogisticRegression | 0.7385 | 0.7426 | 0.9615 | 0.9000 | 0.8642 | 0.9632 |
| RandomForest | 0.7720 | 0.7753 | 0.9612 | 0.8501 | 0.9877 | 0.9724 |
| XGBoost | 0.7633 | 0.7709 | 0.9601 | 0.8289 | 0.9815 | 0.9712 |

## Quy trình

- Dùng duy nhất CSV FedMML hiện có; không sử dụng hoặc tuyên bố sử dụng MIMIC-IV-ED.
- Giữ cùng split bệnh nhân của nghiên cứu trước để so sánh ổn định; không trùng bệnh nhân giữa các tập.
- Tối ưu mỗi thuật toán bằng 6 cấu hình, 3 fold StratifiedGroupKFold; chọn theo Macro F1 CV.
- Điền median, clipping quantile 0.5%/99.5%, StandardScaler và one-hot được fit trong từng fold train.
- Mỗi thuật toán được refit trên train; validation dùng để chọn thuật toán cuối. Không refit bằng test.
- Patient ID, encounter ID, target và văn bản tự do bị loại khỏi đầu vào mô hình.
- SHA-256 của dữ liệu và cả ba artifact được xác minh trước khi mở test; lưu receipt để không tự đánh giá lại.

## Giới hạn

FedMML này có 87.234 lượt khám nhưng chỉ 96 bệnh nhân. Các lượt khám lặp không phải 87.234 mẫu bệnh nhân độc lập.
Dữ liệu đã được khám phá và tập test đã được đánh giá ở nghiên cứu trước; đây là benchmark phát triển lại, không phải holdout mới chưa từng xem.
Xác suất chưa được calibration. Kết quả là nghiên cứu trên FedMML, chưa chứng minh khả năng tổng quát hóa trên bệnh nhân hoặc bệnh viện mới.

## Tệp kết quả

- Thư mục run: outputs/experiments/71eafec26795d19a
- model_lock.json; evaluation_receipt.json; test_metrics.json; test_comparison.csv.
- models/: ba pipeline và các SHA-256 trong manifest.
- figures/: so sánh metric, ma trận nhầm lẫn, ROC từng lớp.
- notebooks/01_preprocessing.ipynb và notebooks/02_model_comparison.ipynb có output dưới từng cell.

## Thử nghiệm cân bằng bổ sung

Phương án cuối chọn bằng validation: **XGBoost / oversample_5_to_4**. Test Macro F1 = **0.77083**; thay đổi **-0.115 điểm phần trăm** so với baseline cuối. Chi tiết và mọi phương án: [BALANCING_RESULTS.md](BALANCING_RESULTS.md).
