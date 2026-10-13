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
5. Chọn **Nhập thủ công**: các trường số bắt đầu trống. Nhập tuổi 18–100 và đủ sáu sinh hiệu đo được (hai huyết áp, nhịp tim, nhịp thở, nhiệt độ, SpO₂); điểm đau tùy chọn. **Điền ca minh họa** lấy dữ liệu tổng hợp từ train. Huyết áp tâm thu phải lớn hơn tâm trương. Đây là quy tắc đầy đủ/nhất quán của demo, không phải quy trình phân luồng đã xác nhận lâm sàng.
6. Kết thúc bằng mức đánh đổi: nhận diện ESI 5 tốt hơn, nhưng ESI 4 giảm. Không gọi điểm đầu ra là độ tin cậy đã được kiểm chứng.

## Kết quả và phạm vi

Mặc định: XGBoost tăng mẫu ESI 5, được chọn bằng validation. Test Macro F1 0,77083, Macro Recall 0,77176, ROC-AUC macro OvR 0,96087, Accuracy 0,84320.

Random Forest baseline: test Macro F1 0,77198. XGBoost cân bằng nâng Recall ESI 5 từ 24,16% lên 32,52%, nhưng giảm Recall ESI 4 từ 74,53% xuống 61,85%. Precision ESI 5 của mô hình mới chỉ 18,17%.

FedMML có 87.234 ca tổng hợp với 96 `patient_id` mô phỏng; test có 13.061 ca/23 ID và đã được sử dụng trong lịch sử dự án. Tuổi và giới tính không nhất quán trong nhiều ID; chia tập không trùng ID chưa đủ chứng minh tính độc lập giữa bệnh nhân. Xác suất chưa hiệu chỉnh và chưa có xác nhận trên bệnh viện/dữ liệu ngoài.

Report ESI 1–2 chỉ mô tả ưu tiên đánh giá cấp cứu theo nhãn dự đoán; bác sĩ xác nhận hướng xử trí, hồi sức hoặc nhập viện. Mô hình không có biến đích nhập viện. Đầu vào thiếu chỉ số sử dụng giá trị điền của pipeline, không tương đương một mô hình đã được đánh giá riêng cho thời điểm phân luồng.

## Cải tiến demo 2.1

Ca minh họa được chọn theo thứ tự ID lượt khám trong train, có đủ sinh hiệu và huyết áp nhất quán; không chọn theo việc mô hình dự đoán đúng. Kết quả lưu thời điểm, phiên bản, toàn bộ đầu vào, các trường được điền giá trị và trạng thái xem xét. Thay đổi trường nhập sẽ xóa kết quả và bản xuất cũ, cần bấm dự đoán lại. Chỉ số nằm trong schema kỹ thuật nhưng ngoài khoảng quan sát của train được đánh dấu cần xem xét; khoảng train không phải ngưỡng lâm sàng.

**Chưa đủ cơ sở kết luận** xuất hiện khi mô hình dùng xét nghiệm nhưng thiếu kết quả, thiếu thông tin phân loại hoặc hai điểm đứng đầu quá gần nhau. Ngưỡng chênh điểm được chọn riêng cho từng mô hình trên validation, với độ phủ tổng thể và độ phủ ESI 1–2 ít nhất 80%; ưu tiên giảm lỗi ESI 1–2 bị hạ xuống 3–5. Đây là tín hiệu xem xét cho nghiên cứu, không phải xác suất đã hiệu chỉnh hoặc ngưỡng an toàn lâm sàng.

Nguồn chưa cung cấp đơn vị xét nghiệm có thể xác minh. App khóa nhập xét nghiệm tùy ý; **Điền ca minh họa** chỉ hiển thị lại giá trị gốc FedMML. Không suy đoán đơn vị hoặc chuyển đổi các giá trị theo giả định.

Bổ sung **RandomForest · không xét nghiệm**, fit lại toàn bộ preprocessing và classifier trên 61.049 ca train với siêu tham số baseline đã khóa. Bốn ứng viên được so sánh trên cùng 11.982 ca validation đáp ứng điều kiện đầu vào; 1.142 ca không đáp ứng được tách khỏi phạm vi này. Lựa chọn ưu tiên tỷ lệ ESI 1–2 → 3–5, rồi Macro F1 và Macro Recall. So với XGBoost mặc định khi bỏ xét nghiệm trên cùng tập này, Macro F1 **0,76455 → 0,77007**, số lỗi ESI 1–2 → 3–5 **106 → 66**. Mô hình mới đạt điều kiện thêm vào danh sách lựa chọn; mô hình mặc định cũ và các kết quả test lịch sử được giữ nguyên.

Thử nghiệm không mở test để chọn mô hình/ngưỡng. Validation đã được dùng trong lịch sử dự án; chưa có test độc lập mới hoặc đánh giá bệnh viện. Bản mới bỏ xét nghiệm nhưng vẫn dùng các trường hành chính và đặc trưng thời gian. Protocol và kết quả tái lập ở `outputs/demo_review/acc23f192f120220` trong dự án nghiên cứu, tạo bằng `tools/evaluate_demo_updates.py`; tóm tắt được nhúng trong registry của demo.

## Nguồn và ghi công

Dataset: [FedMML Emergency Department Triage Dataset](https://huggingface.co/datasets/olaflaitinen/fedmml-ed-triage), DOI `10.57967/hf/7825`. Người quản lý theo dataset card: Gustav Olaf Yunus Laitinen-Fredriksson Lundström-Imanov và Sila Burde Dulger. Dữ liệu được công bố theo [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Dự án chuyển dữ liệu nguồn thành đặc trưng số/phân loại, chia tập theo ID mô phỏng, huấn luyện các pipeline và cung cấp năm ca mẫu train đã bỏ ID/văn bản tự do. Ghi công này không hàm ý tác giả dataset bảo trợ dự án. Bộ CSV đầy đủ không nằm trong gói deploy.

## Đóng gói và kiểm chứng

Gói ZIP chứa app, mã để nạp pipeline, năm artifact mô hình (ba baseline, XGBoost đã chọn và Random Forest không xét nghiệm), năm ca mẫu train không có ID/văn bản tự do, kết quả thực nghiệm, hướng dẫn và manifest SHA-256. Bộ dữ liệu đầy đủ và môi trường `.venv` không cần thiết cho suy luận nên được giữ tại dự án nghiên cứu.

Chạy `python verify_bundle.py` trong thư mục đã giải nén để kiểm tra checksum trước khi mở demo. App kiểm tra phiên bản thư viện và checksum mô hình trước nạp; lỗi artifact sẽ dừng dự đoán, không tự chuyển sang mô hình khác.

Trong dự án nghiên cứu, tạo lại artifact demo bằng `tools/prepare_demo.py`, rồi đóng gói bằng `tools/package_demo.py`.

Nguồn API: [Streamlit text input/callback](https://docs.streamlit.io/develop/api-reference/widgets/st.text_input), [Streamlit AppTest](https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest).

## Deploy miễn phí trên Streamlit Community Cloud

Tại [Streamlit Community Cloud](https://share.streamlit.io/), chọn repo này,
branch `main`, entrypoint **`demo/app.py`**. Trong Advanced settings, chọn
**Python 3.11**. Không cần secrets hoặc API key. Bấm Deploy.

Dependencies được khóa trong `demo/requirements.txt`; `packages.txt` khai báo
thư viện Linux cho XGBoost. Workflow GitHub kiểm tra mô hình và giao diện trên Linux.
Streamlit Community Cloud có thể cho ứng dụng ngủ sau thời gian không hoạt động;
người mở trang có thể đánh thức ứng dụng.

Repo này chỉ chứa bản chạy demo, các mô hình, năm ca tổng hợp và tài liệu kết quả.
Bản nghiên cứu đầy đủ, CSV, notebook, `.venv` và tệp cấu hình tài khoản được giữ tại máy.

Kiểm tra bản tải về: `python verify_bundle.py`.
Chạy trên Linux: `python -m pip install -r demo/requirements.txt`, rồi
`python -m streamlit run demo/app.py`.
