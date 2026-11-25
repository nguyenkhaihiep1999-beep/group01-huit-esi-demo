# GROUP01 HUIT — Demo phân loại ESI (Phiên bản 2.2 · Đồng bộ OOF CV)

Gói demo dùng 5 pipeline đã huấn luyện, tối ưu ngưỡng hoàn toàn trên **Train Out-of-Fold (OOF) CV** và kiểm chứng SHA-256. Không cần huấn luyện lại để trình bày.

**Dữ liệu hoàn toàn tổng hợp, không có bệnh nhân thật.** Đây là demo đồ án/nghiên cứu phân loại ESI 1–5, chưa được xác nhận để chăm sóc bệnh nhân, chẩn đoán bệnh hoặc quyết định nhập viện.

## Chạy trên Windows

Yêu cầu Python 3.11 bản 64-bit. Tại thư mục dự án hoặc thư mục đã giải nén:

    python -m venv .venv
    .venv\Scripts\python.exe -m pip install -r demo/requirements.txt
    .venv\Scripts\python.exe -m streamlit run demo/app.py

Hoặc chạy `run_demo.ps1` rồi mở `http://127.0.0.1:8501`. Dừng bằng `Ctrl+C`.

## Các mô hình trong Demo 2.2 (`run=fd81afda6ebcb7ea`)

1. **XGBoost · OOF anti-overfit (Mặc định — chọn khách quan trên Validation):** Test Macro F1 **0,7711**, Macro Recall **0,7673**, ROC-AUC macro OvR **0,9602**, Accuracy **0,8579**, Recall ESI 1 **99,38%** (0 ca ESI 1–2 bị hạ xuống ESI 4–5).
2. **Soft Voting · RF + XGBoost (`rf_weight = 0,25`):** Test Macro F1 **0,7742**, Accuracy **0,8666**, ổn định nhất qua 10 lần chia nhóm độc lập (**0,7722 ± 0,0021**).
3. **Random Forest · OOF anti-overfit:** Test Macro F1 **0,7700**, Recall ESI 5 **22,00%**.
4. **Logistic Regression · OOF anti-overfit:** Test Accuracy **0,9024**, ROC-AUC **0,9618**, Macro F1 **0,7466**.
5. **XGBoost · không xét nghiệm (`study=6bd3b9114c4adb75`):** Loại bỏ hoàn toàn 11 biến xét nghiệm và 3 cờ/biến đếm phụ thuộc xét nghiệm, tối ưu ngưỡng trên Train OOF CV. Trên 11.982 ca validation hợp lệ đạt **Macro F1 = 0,7789** và giảm lỗi hạ bậc nặng ESI 1–2 $\rightarrow$ 3–5 từ **141 xuống 115 ca** so với để trống xét nghiệm trên mô hình đầy đủ. Chưa mở tập test độc lập cho phiên bản này.

## Cơ chế an toàn & Kiểm chứng của Demo 2.2

- **Nhập thủ công:** Yêu cầu tuổi 18–100, đủ 6 sinh hiệu đo được và huyết áp tâm thu > tâm trương; khóa nhập xét nghiệm tùy ý vì nguồn không xác nhận đơn vị.
- **Tải tệp CSV (Chấm nhiều ca):** Hỗ trợ tối đa 500 dòng mỗi lần, tự động kiểm tra hợp lệ từng dòng theo cùng quy tắc nhập đơn ca, tự bỏ qua cột định danh/nhãn/văn bản tự do (`FORBIDDEN`), cung cấp tệp CSV mẫu 5 ca và xuất bảng kết quả `.csv`.
- **Trạng thái *Chưa đủ cơ sở kết luận*:** Tự động bật khi thiếu xét nghiệm (với mô hình dùng xét nghiệm), chỉ số ngoài khoảng quan sát tập train, quy tắc ngưỡng OOF chọn nhãn khác mức có điểm thô cao nhất (`decision_rule_shifted`), hoặc chênh lệch 2 mức điểm thô dẫn đầu nhỏ hơn ngưỡng validation (`review_policy`).
- **Xóa kết quả cũ tức thì:** Mọi chỉnh sửa đầu vào đều xóa kết quả và bản tải JSON cũ để tránh nhầm lẫn.
- **Kiểm chứng toàn vẹn:** Chạy `python verify_bundle.py` trước khi mở demo. App tự kiểm tra phiên bản thư viện và SHA-256 trước khi nạp bất kỳ tệp `.joblib` nào.

## Nguồn và ghi công

Dataset: [FedMML Emergency Department Triage Dataset](https://huggingface.co/datasets/olaflaitinen/fedmml-ed-triage), DOI `10.57967/hf/7825` (Gustav Olaf Yunus Laitinen-Fredriksson Lundström-Imanov & Sila Burde Dulger), phát hành theo [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

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
