# GROUP01 HUIT — demo phân loại ESI

Gói demo dùng các pipeline đã huấn luyện và được kiểm chứng SHA-256. Không cần huấn luyện lại để trình bày.

**Dữ liệu hoàn toàn tổng hợp, không có bệnh nhân thật.** Đây là demo đồ án/nghiên cứu phân loại ESI 1–5, chưa được xác nhận để chăm sóc bệnh nhân, chẩn đoán bệnh hoặc quyết định nhập viện. Dấu hiệu sinh tồn và xét nghiệm được nguồn sinh theo mức ESI; điểm số đo khả năng học quy luật của dữ liệu sinh.

## Chạy trên Windows

Yêu cầu Python 3.11 bản 64-bit. Tại thư mục dự án hoặc thư mục đã giải nén:

    python -m venv .venv
    .venv\Scripts\python.exe -m pip install -r demo/requirements.txt
    .venv\Scripts\python.exe -m streamlit run demo/app.py

Hoặc chạy `run_demo.ps1`. Script tự chọn `.venv` đã có; nếu chưa có thì tạo môi trường và cài dependencies. Cài lần đầu cần Internet. Các lần mở demo dùng mô hình có sẵn, không cần tải dữ liệu hoặc huấn luyện.

Mở http://127.0.0.1:8501. Dừng bằng Ctrl+C trong terminal đang chạy. Gói chỉ lắng nghe trên loopback máy đang dùng.

## Trình bày trong khoảng 5 phút

1. Giới thiệu nguồn FedMML, bài toán ESI 1–5 và ba thuật toán.
2. Mở **Kết quả thực nghiệm**: Macro F1, Macro Recall, ROC-AUC macro OvR và ma trận nhầm lẫn.
3. Quay lại **Dự đoán một ca**, chọn một ca mẫu rồi bấm **Dự đoán ca mẫu**. Ca mẫu lấy từ train để minh họa thao tác, không dùng làm bằng chứng test.
4. Xem dự đoán và điểm đầu ra trên năm lớp; so sánh cùng một ca với ba thuật toán và biến thể XGBoost cân bằng.
5. Chọn **Nhập thủ công**, nhập tuổi 18–100 và ít nhất một dấu hiệu sinh tồn; để trống xét nghiệm chưa có. Pipeline đã lưu tự xử lý null, clipping, chuẩn hóa và mã hóa. Đơn vị xét nghiệm chưa được nguồn xác nhận đầy đủ, chỉ nhập giá trị mô phỏng theo ca FedMML.
6. Kết thúc bằng mức đánh đổi: nhận diện ESI 5 tốt hơn, nhưng ESI 4 giảm. Không gọi điểm đầu ra là độ tin cậy đã được kiểm chứng.

## Kết quả và phạm vi

Mặc định: XGBoost tăng mẫu ESI 5, được chọn bằng validation. Test Macro F1 0,77083, Macro Recall 0,77176, ROC-AUC macro OvR 0,96087, Accuracy 0,84320.

Random Forest baseline: test Macro F1 0,77198. XGBoost cân bằng nâng Recall ESI 5 từ 24,16% lên 32,52%, nhưng giảm Recall ESI 4 từ 74,53% xuống 61,85%. Precision ESI 5 của mô hình mới chỉ 18,17%.

FedMML có 87.234 ca tổng hợp với 96 `patient_id` mô phỏng; test có 13.061 ca/23 ID và đã được sử dụng trong lịch sử dự án. Tuổi và giới tính không nhất quán trong nhiều ID; chia tập không trùng ID chưa đủ chứng minh tính độc lập giữa bệnh nhân. Xác suất chưa hiệu chỉnh và chưa có xác nhận trên bệnh viện/dữ liệu ngoài.

Report ESI 1–2 chỉ mô tả ưu tiên đánh giá cấp cứu theo nhãn dự đoán; bác sĩ xác nhận hướng xử trí, hồi sức hoặc nhập viện. Mô hình không có biến đích nhập viện. Đầu vào thiếu chỉ số sử dụng giá trị điền của pipeline, không tương đương một mô hình đã được đánh giá riêng cho thời điểm phân luồng.

## Nguồn và ghi công

Dataset: [FedMML Emergency Department Triage Dataset](https://huggingface.co/datasets/olaflaitinen/fedmml-ed-triage), DOI `10.57967/hf/7825`. Người quản lý theo dataset card: Gustav Olaf Yunus Laitinen-Fredriksson Lundström-Imanov và Sila Burde Dulger. Dữ liệu được công bố theo [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Dự án chuyển dữ liệu nguồn thành đặc trưng số/phân loại, chia tập theo ID mô phỏng, huấn luyện bốn pipeline và cung cấp năm ca mẫu train đã bỏ ID/văn bản tự do. Ghi công này không hàm ý tác giả dataset bảo trợ dự án. Bộ CSV đầy đủ không nằm trong gói deploy.

## Đóng gói và kiểm chứng

Gói ZIP chứa app, mã để nạp pipeline, bốn artifact mô hình (ba baseline và XGBoost đã chọn), năm ca mẫu train không có ID/văn bản tự do, kết quả thực nghiệm, hướng dẫn và manifest SHA-256. Bộ dữ liệu đầy đủ và môi trường `.venv` không cần thiết cho suy luận nên được giữ tại dự án nghiên cứu.

Chạy `python verify_bundle.py` trong thư mục đã giải nén để kiểm tra checksum trước khi mở demo. App kiểm tra phiên bản thư viện và checksum mô hình trước nạp; lỗi artifact sẽ dừng dự đoán, không tự chuyển sang mô hình khác.

Trong dự án nghiên cứu, tạo lại artifact demo bằng `tools/prepare_demo.py`, rồi đóng gói bằng `tools/package_demo.py`.

Nguồn API: [Streamlit forms](https://docs.streamlit.io/develop/api-reference/execution-flow/st.form), [Streamlit AppTest](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest).
