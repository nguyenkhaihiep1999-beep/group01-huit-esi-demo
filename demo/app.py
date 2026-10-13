"""Streamlit presentation app using the original fitted ESI pipelines."""

from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from demo.inference import ARTIFACTS, predict_case, read_registry
from demo.reporting import emergency_report
from demo.policy import assess_prediction

st.set_page_config(page_title="Phân loại ESI · FedMML", page_icon=None, layout="wide")
st.html(f"<style>{Path(__file__).with_name('styles.css').read_text(encoding='utf-8')}</style>")

VITAL_LABELS = {
    "systolic_bp": "Huyết áp tâm thu (mmHg)", "diastolic_bp": "Huyết áp tâm trương (mmHg)",
    "heart_rate": "Nhịp tim (lần/phút)", "respiratory_rate": "Nhịp thở (lần/phút)",
    "temperature": "Nhiệt độ (°C)", "spo2": "SpO₂ (%)", "pain_score": "Điểm đau (0–10)",
}
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
                                           "payload": dict(payload), "example": example, "input_mode": mode,
                                           "created_at_utc": datetime.now(timezone.utc).isoformat(),
                                           "demo_version": registry.get("demo_version")}
    except (OSError, ValueError) as exc:
        st.session_state.pop("prediction", None)
        message = str(exc)
        for field, label in FIELD_LABELS.items():
            if message.startswith(field + ":"):
                message = label + message[len(field):]
                break
        st.error(message.replace("giá trị hợp lệ của schema nằm trong", "vui lòng nhập giá trị trong khoảng"))


def clear_prediction():
    st.session_state.pop("prediction", None)


def fill_manual_example():
    clear_prediction()
    case = next(case for case in registry["examples"] if case["id"] == st.session_state["manual_example_id"])
    for field, value in case["input"].items():
        if field == "arrival_timestamp":
            arrived = datetime.fromisoformat(value)
            st.session_state["arrival_date"] = arrived.date()
            st.session_state["arrival_time"] = arrived.time()
        elif field in ["sex", "country", "site_id"]:
            st.session_state[field] = value
        else:
            st.session_state[field] = "" if value is None else str(value)
    st.session_state["manual_sample_origin"] = case["id"]


def clear_manual_inputs():
    clear_prediction()
    st.session_state.pop("manual_sample_origin", None)
    for field in ["age", *VITAL_LABELS, *LAB_LABELS]:
        st.session_state[field] = ""
    for field in ["sex", "country", "site_id"]:
        st.session_state[field] = None
    now = datetime.now(ZoneInfo("Asia/Ho_Chi_Minh"))
    st.session_state["arrival_date"] = now.date()
    st.session_state["arrival_time"] = now.time().replace(second=0, microsecond=0, tzinfo=None)


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
        with st.container(border=True, key="manual_case"):
            st.markdown("### Thông tin ca khám")
            st.caption("Tuổi 18–100 và đủ sáu sinh hiệu đo được. Điểm đau có thể để trống. Đây là quy tắc đầu vào của demo nghiên cứu.")
            example_ids = [case["id"] for case in registry["examples"]]
            st.selectbox("Ca minh họa để điền", example_ids,
                         format_func=lambda key: f"Ca {example_ids.index(key) + 1:02d} · dữ liệu tổng hợp ESI {example_ids.index(key) + 1}",
                         key="manual_example_id")
            load_col, clear_col = st.columns(2)
            load_col.button("Điền ca minh họa", key="fill_manual_example", on_click=fill_manual_example)
            clear_col.button("Xóa dữ liệu", key="clear_manual_inputs", on_click=clear_manual_inputs)
            if st.session_state.get("manual_sample_origin"):
                st.caption("Đã điền dữ liệu mô phỏng từ ca train; xét nghiệm giữ giá trị gốc và chỉ được xem.")
            cols = st.columns(4)
            payload = {"age": cols[0].text_input("Tuổi (năm)", key="age", on_change=clear_prediction),
                       "sex": cols[1].selectbox("Giới tính", registry["categories"]["sex"],
                                               index=None, placeholder="Chưa chọn",
                                               format_func=lambda value: {"F": "Nữ", "M": "Nam"}.get(value, value), key="sex", on_change=clear_prediction),
                       "country": cols[2].selectbox("Quốc gia", registry["categories"]["country"], index=None,
                                                   placeholder="Chưa chọn", key="country", on_change=clear_prediction),
                       "site_id": cols[3].selectbox("Cơ sở", registry["categories"]["site_id"], index=None,
                                                   placeholder="Chưa chọn", key="site_id", on_change=clear_prediction)}
            cols = st.columns(2)
            now = datetime.now(ZoneInfo("Asia/Ho_Chi_Minh"))
            if "arrival_date" not in st.session_state:
                st.session_state["arrival_date"] = now.date()
            if "arrival_time" not in st.session_state:
                st.session_state["arrival_time"] = now.time().replace(second=0, microsecond=0, tzinfo=None)
            arrival_date = cols[0].date_input("Ngày tiếp nhận", value=None, key="arrival_date", on_change=clear_prediction)
            arrival_time = cols[1].time_input("Giờ tiếp nhận", value=None,
                                            key="arrival_time", on_change=clear_prediction)
            payload["arrival_timestamp"] = datetime.combine(arrival_date, arrival_time).isoformat() if arrival_date and arrival_time else None
            st.markdown("#### Dấu hiệu sinh tồn")
            cols = st.columns(4)
            for i, (field, label) in enumerate(VITAL_LABELS.items()):
                payload[field] = cols[i % 4].text_input(label, key=field, on_change=clear_prediction)
            with st.expander("Bổ sung xét nghiệm"):
                st.caption("Chưa xác minh được đơn vị từ nguồn. Chỉ xem giá trị gốc khi điền ca minh họa; nhập xét nghiệm thủ công đang khóa.")
                cols = st.columns(3)
                for i, (field, label) in enumerate(LAB_LABELS.items()):
                    payload[field] = cols[i % 3].text_input(label, key=field, disabled=True)
            submitted = st.button("Xem dự đoán", type="primary", key="predict_manual")
        if submitted:
            perform_prediction(payload)

    saved = st.session_state.get("prediction")
    current_case_id = example_id if mode == "Ca mẫu FedMML" else None
    if saved and (
        saved["primary"]["model_id"] != model_id
        or saved.get("demo_version") != registry.get("demo_version")
        or saved.get("input_mode") != mode
        or (mode == "Nhập thủ công" and saved.get("payload") != payload)
        or (mode == "Ca mẫu FedMML" and (not saved.get("example") or saved["example"]["id"] != current_case_id))
    ):
        st.session_state.pop("prediction", None)
        saved = None

    if saved:
        result = saved["primary"]
        saved["emergency_report"] = emergency_report(result["predicted_esi"])
        saved["review"] = assess_prediction(result, registry["models"][result["model_id"]])
        with st.container(border=True, key="prediction_result"):
            st.markdown("### Kết quả dự đoán")
            created = datetime.fromisoformat(saved["created_at_utc"]).astimezone(ZoneInfo("Asia/Ho_Chi_Minh"))
            st.caption(f"Lần dự đoán: {created:%d/%m/%Y %H:%M:%S} (GMT+7) · bản {saved['demo_version']}")
            show_case_summary(saved["payload"])
            review = saved["review"]
            if review["status"] == "review_required":
                st.warning(review["label"] + ": " + " ".join(review["reasons"]))
            else:
                st.caption(review["label"] + " · chưa có xác nhận lâm sàng.")
            left, right = st.columns([1, 1.6], gap="large")
            with left:
                st.metric("ESI dự đoán", f"ESI {result['predicted_esi']}")
                st.caption(model_name(result["model_id"]))
                if saved["example"]:
                    st.write(f"Nhãn tham khảo: **ESI {saved['example']['observed_esi']}**")
                if result["missing_fields"]:
                    names = [FIELD_LABELS.get(field, field) for field in result["missing_fields"]]
                    st.caption("Trường được mô hình điền giá trị: " + ", ".join(names) + ".")
                    st.caption("Đầu vào thiếu chỉ số: kết quả phụ thuộc giá trị điền của mô hình; chưa được xác nhận lâm sàng.")
            with right:
                st.markdown("**Điểm đầu ra theo mức ESI**")
                show_scores(result)
                st.caption("Xác suất chưa hiệu chỉnh; không thể dùng như độ tin cậy lâm sàng.")
                if review["minimum_margin"] is not None:
                    st.caption(f"Chênh hai mức đứng đầu: {review['score_margin'] * 100:.2f} điểm %. Ngưỡng xem xét từ validation: {review['minimum_margin'] * 100:.0f} điểm %; không phải ngưỡng an toàn lâm sàng.")
            report = saved["emergency_report"]
            if report:
                with st.container(key=f"emergency_report_esi_{report['esi_level']}"):
                    st.markdown(f"**Khuyến nghị theo ESI dự đoán · {report['priority']}:** {report['recommendation']}")
            else:
                st.caption("Dự đoán ESI 3–5 không được dùng để loại trừ tình huống cấp cứu.")
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
    record = registry["models"][model_id]
    on_test = record["test_metrics"] is not None
    metrics = record["test_metrics"] if on_test else record["validation_metrics"]
    st.markdown("### Kết quả trên tập test" if on_test else "### Kết quả trên validation")
    if on_test:
        st.caption(f"{registry['test_rows']:,}".replace(",", ".") + f" ca tổng hợp · {registry['test_patients']} ID mô phỏng")
    else:
        st.caption(f"{record['validation_rows']:,}".replace(",", ".") + " ca validation đáp ứng điều kiện đầu vào. Chưa có test độc lập cho phiên bản này.")
    st.markdown(f"**{model_name(model_id)}**")
    with st.container(key="test_metrics"):
        cols = st.columns(4)
        for col, label, field in zip(cols, ["Macro F1", "Macro Recall", "ROC-AUC · macro OvR", "Accuracy"],
                                     ["macro_f1", "macro_recall", "roc_auc_macro_ovr", "accuracy"]):
            col.metric(label, f"{metrics[field]:.4f}")
    st.caption("Tập test đã được sử dụng trong quá trình phát triển dự án." if on_test
               else "Validation dùng để chọn mô hình và ngưỡng xem xét; không phải đánh giá độc lập cuối cùng.")
    severe_rows = sum(sum(row) for row in metrics["confusion_matrix"][:2])
    severe_errors = sum(sum(row[2:]) for row in metrics["confusion_matrix"][:2])
    risk_cols = st.columns(2)
    risk_cols[0].metric("ESI 1–2 dự đoán thành 3–5", f"{severe_errors}/{severe_rows}")
    risk_cols[1].metric("Precision ESI 5", f"{metrics['per_class']['5']['precision'] * 100:.2f}%")
    with st.expander("Đối chiếu bốn mô hình trên test lịch sử", expanded=True):
        rows = [{"Mô hình": model_name(key), "Macro F1": item["test_metrics"]["macro_f1"],
                 "Macro Recall": item["test_metrics"]["macro_recall"], "ROC-AUC": item["test_metrics"]["roc_auc_macro_ovr"],
                 "Recall ESI 4": item["test_metrics"]["per_class"]["4"]["recall"],
                 "Recall ESI 5": item["test_metrics"]["per_class"]["5"]["recall"]}
                for key, item in registry["models"].items() if item["test_metrics"] is not None]
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
    study = registry.get("review_study")
    if study:
        with st.expander("Thử nghiệm mô hình không xét nghiệm"):
            st.caption(f"Cùng {study['validation_rows']:,} ca validation có đủ sáu sinh hiệu và huyết áp nhất quán; {study['excluded_validation_rows']:,} ca không đáp ứng điều kiện được loại khỏi phạm vi đánh giá này.".replace(",", "."))
            rows = []
            for key, candidate in study["candidates"].items():
                value = candidate["validation_metrics"]
                rows.append({"Phiên bản": MODEL_NAMES.get(key, key) + " · bỏ xét nghiệm",
                             "Macro F1": value["macro_f1"], "Recall ESI 1": value["per_class"]["1"]["recall"],
                             "Recall ESI 2": value["per_class"]["2"]["recall"], "ESI 1–2 → 3–5": value["severe_to_3_5"]})
            st.dataframe(pd.DataFrame(rows).round(4), hide_index=True, width="stretch")
            reference = study["masked_reference"]
            st.caption(f"Mô hình đầy đủ khi để trống xét nghiệm trên cùng validation: Macro F1 {reference['macro_f1']:.4f}; {reference['severe_to_3_5']} ca ESI 1–2 → 3–5.")
            st.write("Đủ điều kiện đưa phiên bản không xét nghiệm vào danh sách mô hình." if study["promoted_on_validation"]
                     else "Chưa đạt điều kiện cải thiện trên validation; phiên bản không xét nghiệm chưa được đưa vào dự đoán.")
            st.caption("Siêu tham số giữ theo các baseline đã khóa; preprocessing và mô hình được fit lại trên train. Không mở test để chọn phiên bản này. Validation đã dùng trong lịch sử và dữ liệu vẫn hoàn toàn tổng hợp.")

st.html(f'<div class="page-footer">FedMML · Nghiên cứu ESI <span>Demo nghiên cứu · {escape(registry.get("demo_version", ""))}</span></div>')
