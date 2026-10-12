"""Streamlit presentation app using the original fitted ESI pipelines."""

from datetime import date, datetime, time
from html import escape
import json
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from demo.inference import ARTIFACTS, predict_case, read_registry
from demo.reporting import emergency_report

st.set_page_config(page_title="Phân loại ESI · FedMML", page_icon=None, layout="wide")
st.html(f"<style>{Path(__file__).with_name('styles.css').read_text(encoding='utf-8')}</style>")

VITAL_LABELS = {
    "systolic_bp": "Huyết áp tâm thu (mmHg)", "diastolic_bp": "Huyết áp tâm trương (mmHg)",
    "heart_rate": "Nhịp tim (lần/phút)", "respiratory_rate": "Nhịp thở (lần/phút)",
    "temperature": "Nhiệt độ (°C)", "spo2": "SpO₂ (%)", "pain_score": "Điểm đau (0–10)",
}
DEFAULT_VITALS = {"systolic_bp": "120", "diastolic_bp": "80", "heart_rate": "75", "respiratory_rate": "16",
                  "temperature": "37", "spo2": "98", "pain_score": "2"}
LAB_LABELS = {"wbc": "WBC", "hemoglobin": "Hemoglobin", "platelet_count": "Tiểu cầu", "sodium": "Natri",
              "potassium": "Kali", "creatinine": "Creatinine", "glucose": "Glucose", "troponin": "Troponin",
              "bnp": "BNP", "lactate": "Lactate", "inr": "INR"}
FIELD_LABELS = {"age": "Tuổi", "sex": "Giới tính", "country": "Quốc gia", "site_id": "Cơ sở",
                "arrival_timestamp": "Thời điểm tiếp nhận", **VITAL_LABELS, **LAB_LABELS}
MODEL_NAMES = {
    "selected_balancing": "XGBoost · tăng mẫu ESI 5",
    "baseline_LogisticRegression": "Logistic Regression",
    "baseline_RandomForest": "Random Forest",
    "baseline_XGBoost": "XGBoost · bản gốc",
}


def model_name(key):
    return MODEL_NAMES.get(key, registry["models"][key]["label"])


def display_value(value):
    if value is None or value == "":
        return "Chưa có"
    return f"{value:g}" if isinstance(value, (int, float)) else str(value)


def show_case_summary(payload):
    age = escape(display_value(payload.get("age")))
    sex = {"F": "Nữ", "M": "Nam"}.get(payload.get("sex"), display_value(payload.get("sex")))
    st.html(f'<p class="case-person">{age} tuổi <span>· {escape(sex)} · {escape(display_value(payload.get("country")))}</span></p>')
    values = [("Huyết áp", f"{display_value(payload.get('systolic_bp'))} / {display_value(payload.get('diastolic_bp'))}", "mmHg"),
              ("Nhịp tim", display_value(payload.get("heart_rate")), "lần/phút"),
              ("Nhiệt độ", display_value(payload.get("temperature")), "°C"),
              ("SpO₂", display_value(payload.get("spo2")), "%")]
    tiles = "".join(f'<div class="case-vital"><span>{escape(label)}</span><strong>{escape(value)}</strong><small>{escape(unit)}</small></div>'
                    for label, value, unit in values)
    st.html(f'<div class="case-vitals">{tiles}</div>')


def show_scores(result):
    rows = []
    for label in registry["classes"]:
        score = result["scores"][str(label)] * 100
        selected = " selected" if label == result["predicted_esi"] else ""
        rows.append(f'<div class="score-row{selected}"><span>ESI {label}</span>'
                    f'<div class="score-track"><div class="score-fill" style="width:{score:.6f}%"></div></div>'
                    f'<span class="score-value">{score:.2f}%</span></div>')
    st.html('<div class="score-list" aria-label="Điểm đầu ra cho năm mức ESI">' + "".join(rows) + '</div>')


try:
    registry = read_registry()
except (OSError, ValueError) as exc:
    st.error(f"Chưa thể mở mô hình: {exc}")
    st.stop()

with st.sidebar:
    st.html('<div class="sidebar-brand"><span class="group-brand">GROUP01 HUIT</span>'
            'FedMML<span>Nghiên cứu phân loại ESI</span></div>')
    keys = list(registry["models"])
    model_id = st.selectbox("Mô hình", keys, index=keys.index(registry["selected_model"]),
                            format_func=model_name, key="model_id")
    compare = st.checkbox("So sánh các mô hình", value=True, key="compare_models")
    st.caption("XGBoost tăng mẫu ESI 5 được chọn trên tập validation.")
    st.divider()
    st.markdown("**Về bộ dữ liệu**")
    st.caption("87.234 ca tổng hợp\n\n96 ID mô phỏng · 5 mức ESI")
    st.caption("Nguồn: [FedMML · CC BY 4.0](https://huggingface.co/datasets/olaflaitinen/fedmml-ed-triage)")

st.html('<div class="page-kicker">FedMML <span>/</span> Thực nghiệm phân loại cấp cứu</div>')
st.title("Phân loại ESI")
st.write("Chọn một ca mẫu hoặc nhập thông tin để xem dự đoán.")
st.html('<p class="research-note">Dữ liệu hoàn toàn tổng hợp, không có bệnh nhân thật. Demo nghiên cứu; không dùng cho quyết định phân loại cấp cứu hoặc nhập viện.</p>')
prediction_tab, results_tab = st.tabs(["Dự đoán một ca", "Kết quả thực nghiệm"])


def perform_prediction(payload, example=None):
    try:
        primary = predict_case(payload, model_id)
        if compare:
            comparisons = [primary if key == model_id else predict_case(payload, key) for key in keys]
        else:
            comparisons = [primary]
        st.session_state["prediction"] = {"primary": primary, "comparisons": comparisons,
                                           "payload": payload, "example": example, "input_mode": mode}
    except (OSError, ValueError) as exc:
        st.session_state.pop("prediction", None)
        message = str(exc)
        for field, label in FIELD_LABELS.items():
            if message.startswith(field + ":"):
                message = label + message[len(field):]
                break
        st.error(message.replace("giá trị hợp lệ của schema nằm trong", "vui lòng nhập giá trị trong khoảng"))


with prediction_tab:
    mode = st.radio("Nguồn đầu vào", ["Ca mẫu FedMML", "Nhập thủ công"], horizontal=True,
                    key="input_mode", label_visibility="collapsed")
    if mode == "Ca mẫu FedMML":
        with st.container(border=True, key="input_panel"):
            st.markdown("### Ca khám mẫu")
            st.caption("Ca tổng hợp lấy từ tập train, dùng để thử thao tác.")
            examples = {case["id"]: case for case in registry["examples"]}
            example_id = st.selectbox("Chọn ca mẫu", list(examples),
                                     format_func=lambda key: f"Ca {list(examples).index(key) + 1:02d} · nhãn tham khảo ESI {examples[key]['observed_esi']}",
                                     key="example_id")
            case = examples[example_id]
            show_case_summary(case["input"])
            if st.button("Dự đoán ca mẫu", type="primary", key="predict_example"):
                perform_prediction(case["input"], case)
            with st.expander("Thông tin đầy đủ của ca mẫu"):
                st.caption("Nhãn tham khảo không được dùng làm đầu vào dự đoán.")
                st.dataframe(pd.DataFrame({"Thông tin": [FIELD_LABELS.get(field, field) for field in case["input"]],
                                           "Giá trị": [display_value(value) for value in case["input"].values()]}),
                             hide_index=True, width="stretch")
    else:
        with st.form("manual_case", enter_to_submit=False):
            st.markdown("### Thông tin ca khám")
            st.caption("Demo cho tuổi 18–100. Cần ít nhất một sinh hiệu; chỉ số thiếu được mô hình điền giá trị.")
            cols = st.columns(4)
            payload = {"age": cols[0].text_input("Tuổi (năm)", value="40", key="age"),
                       "sex": cols[1].selectbox("Giới tính", registry["categories"]["sex"],
                                               format_func=lambda value: {"F": "Nữ", "M": "Nam"}.get(value, value), key="sex"),
                       "country": cols[2].selectbox("Quốc gia", registry["categories"]["country"], key="country"),
                       "site_id": cols[3].selectbox("Cơ sở", registry["categories"]["site_id"], key="site_id")}
            cols = st.columns(2)
            arrival_date = cols[0].date_input("Ngày tiếp nhận", value=date(2026, 1, 1), key="arrival_date")
            arrival_time = cols[1].time_input("Giờ tiếp nhận", value=time(12, 0), key="arrival_time")
            payload["arrival_timestamp"] = datetime.combine(arrival_date, arrival_time).isoformat()
            st.markdown("#### Dấu hiệu sinh tồn")
            cols = st.columns(4)
            for i, (field, label) in enumerate(VITAL_LABELS.items()):
                payload[field] = cols[i % 4].text_input(label, value=DEFAULT_VITALS[field], key=field)
            with st.expander("Bổ sung xét nghiệm"):
                st.caption("Nguồn chưa xác nhận đầy đủ đơn vị xét nghiệm. Chỉ nhập giá trị mô phỏng theo ca FedMML; để trống nếu chưa có.")
                cols = st.columns(3)
                for i, (field, label) in enumerate(LAB_LABELS.items()):
                    payload[field] = cols[i % 3].text_input(label, value="", key=field)
            submitted = st.form_submit_button("Xem dự đoán", type="primary", key="predict_manual")
        if submitted:
            perform_prediction(payload)

    saved = st.session_state.get("prediction")
    current_case_id = example_id if mode == "Ca mẫu FedMML" else None
    if saved and (
        saved["primary"]["model_id"] != model_id
        or saved.get("input_mode") != mode
        or (mode == "Ca mẫu FedMML" and (not saved.get("example") or saved["example"]["id"] != current_case_id))
    ):
        st.session_state.pop("prediction", None)
        saved = None

    if saved:
        result = saved["primary"]
        saved["emergency_report"] = emergency_report(result["predicted_esi"])
        with st.container(border=True, key="prediction_result"):
            st.markdown("### Kết quả dự đoán")
            left, right = st.columns([1, 1.6], gap="large")
            with left:
                st.metric("ESI dự đoán", f"ESI {result['predicted_esi']}")
                st.caption(model_name(result["model_id"]))
                if saved["example"]:
                    st.write(f"Nhãn tham khảo: **ESI {saved['example']['observed_esi']}**")
                if result["missing_fields"]:
                    st.caption(f"{len(result['missing_fields'])} chỉ số trống đã được xử lý theo mô hình đã lưu.")
                    st.caption("Đầu vào thiếu chỉ số: kết quả phụ thuộc giá trị điền của mô hình; chưa được xác nhận lâm sàng.")
            with right:
                st.markdown("**Điểm đầu ra theo mức ESI**")
                show_scores(result)
                st.caption("Xác suất chưa hiệu chỉnh; không thể dùng như độ tin cậy lâm sàng.")
            report = saved["emergency_report"]
            if report:
                with st.container(key=f"emergency_report_esi_{report['esi_level']}"):
                    st.markdown(f"**Khuyến nghị theo ESI dự đoán · {report['priority']}:** {report['recommendation']}")
        if compare:
            if len(saved.get("comparisons", [])) != len(keys):
                try:
                    saved["comparisons"] = [
                        saved["primary"] if key == saved["primary"]["model_id"] else predict_case(saved["payload"], key)
                        for key in keys
                    ]
                    st.session_state["prediction"] = saved
                except (OSError, ValueError) as exc:
                    st.warning(f"Chưa thể cập nhật so sánh các mô hình: {exc}")
            if len(saved.get("comparisons", [])) > 1:
                st.markdown("#### So sánh trên cùng một ca")
                rows = [{"Mô hình": model_name(item["model_id"]), "ESI dự đoán": item["predicted_esi"],
                         **{f"ESI {c} (%)": item["scores"][str(c)] * 100 for c in registry["classes"]}}
                        for item in saved["comparisons"]]
                st.dataframe(pd.DataFrame(rows).round(2), hide_index=True, width="stretch")
        st.download_button("Lưu kết quả (.json)", json.dumps(saved, ensure_ascii=False, indent=2),
                           file_name="fedmml_prediction.json", mime="application/json", key="download_prediction")

with results_tab:
    st.markdown("### Kết quả trên tập test")
    st.caption(f"{registry['test_rows']:,}".replace(",", ".") + f" ca tổng hợp · {registry['test_patients']} ID mô phỏng")
    record = registry["models"][model_id]
    metrics = record["test_metrics"]
    st.markdown(f"**{model_name(model_id)}**")
    with st.container(key="test_metrics"):
        cols = st.columns(4)
        for col, label, field in zip(cols, ["Macro F1", "Macro Recall", "ROC-AUC · macro OvR", "Accuracy"],
                                     ["macro_f1", "macro_recall", "roc_auc_macro_ovr", "accuracy"]):
            col.metric(label, f"{metrics[field]:.4f}")
    st.caption("Tập test đã được sử dụng trong quá trình phát triển dự án.")
    with st.expander("Đối chiếu các mô hình", expanded=True):
        rows = [{"Mô hình": model_name(key), "Macro F1": item["test_metrics"]["macro_f1"],
                 "Macro Recall": item["test_metrics"]["macro_recall"], "ROC-AUC": item["test_metrics"]["roc_auc_macro_ovr"],
                 "Recall ESI 4": item["test_metrics"]["per_class"]["4"]["recall"],
                 "Recall ESI 5": item["test_metrics"]["per_class"]["5"]["recall"]}
                for key, item in registry["models"].items()]
        st.dataframe(pd.DataFrame(rows).round(4), hide_index=True, width="stretch")
    st.markdown("#### Kết quả từng mức ESI")
    per_class = pd.DataFrame(metrics["per_class"]).T.rename(columns={"precision": "Precision", "recall": "Recall", "f1-score": "F1", "support": "Số ca"})
    per_class.index = [f"ESI {c}" for c in registry["classes"]]
    st.dataframe(per_class.round(4), width="stretch", column_config={"Số ca": st.column_config.NumberColumn(format="%d")})
    st.markdown("#### Ma trận nhầm lẫn")
    st.caption("Hàng là mức ESI thực tế, cột là mức dự đoán. Các ô thể hiện số lượt khám.")
    st.dataframe(pd.DataFrame(metrics["confusion_matrix"], index=[f"Thực tế ESI {c}" for c in registry["classes"]],
                             columns=[f"Dự đoán ESI {c}" for c in registry["classes"]]), width="stretch")
    with st.expander("Đọc thêm về mức đánh đổi"):
        st.image(str(ARTIFACTS / "test_per_class.png"), caption="Random Forest bản gốc và XGBoost tăng mẫu ESI 5.")
        st.write("Recall ESI 5 tăng từ 24,16% lên 32,52%, còn Recall ESI 4 giảm từ 74,53% xuống 61,85%. Macro F1 giảm nhẹ từ 77,20% xuống 77,08%.")
        st.write("Dữ liệu hoàn toàn tổng hợp; 96 patient_id không đại diện cho 96 bệnh nhân thật. Các ID có tuổi và giới tính không nhất quán, nên chia tập không trùng ID chưa chứng minh tính độc lập giữa bệnh nhân.")
        st.write("Dấu hiệu sinh tồn và xét nghiệm được sinh theo nhãn ESI. Kết quả ở đây đo mức học quy luật dữ liệu sinh; chưa chứng minh hiệu quả tại bệnh viện hoặc khả năng quyết định nhập viện.")

st.html('<div class="page-footer">FedMML · Nghiên cứu ESI <span>Demo nghiên cứu</span></div>')
