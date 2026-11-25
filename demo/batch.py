"""Batch (CSV) scoring for the Streamlit demo using the same checksum-verified pipelines.

Each row is validated with exactly the same rules as a single case (``validate_input``).
The model file is verified (SHA-256) and loaded once per batch, then all valid rows are
scored together. Forbidden columns (IDs, target, free text) are never used as inputs;
``encounter_id`` is kept only as a row identifier and ``esi_level`` only as a reference label.
"""

import io
import math

import numpy as np
import pandas as pd

from demo.inference import ARTIFACTS, INPUT_FIELDS, _load_verified, read_registry, validate_input, verify_model
from demo.policy import LAB_FIELDS, assess_prediction
from demo.reporting import emergency_report
from fedmml_triage.config import CLASSES, FORBIDDEN, TARGET
from fedmml_triage.data import make_features

MAX_BATCH_ROWS = 500
MAX_BATCH_BYTES = 5 * 1024 * 1024
ID_COLUMN = "encounter_id"
REQUIRED_COLUMNS = ("age", "systolic_bp", "diastolic_bp", "heart_rate",
                    "respiratory_rate", "temperature", "spo2", "arrival_timestamp")
STRIPPED_COLUMNS = sorted(set(FORBIDDEN) - {"arrival_timestamp"})


def _cell(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        value = value.strip()
        return value or None
    return value.item() if hasattr(value, "item") else value


def _parse_reference_esi(value):
    if value is None or isinstance(value, (bool, np.bool_)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isfinite(number) and number.is_integer() and int(number) in CLASSES:
        return int(number)
    return None


def read_batch_csv(data, max_rows=MAX_BATCH_ROWS, max_bytes=MAX_BATCH_BYTES):
    """Parse uploaded CSV bytes into a DataFrame with basic size and header checks."""
    if data is None or not bytes(data).strip():
        raise ValueError("Tệp CSV rỗng; vui lòng chọn tệp có ít nhất 1 dòng dữ liệu.")
    if len(data) > max_bytes:
        raise ValueError(f"Tệp quá lớn (tối đa {max_bytes // (1024 * 1024)} MB).")
    try:
        preview = pd.read_csv(io.BytesIO(data), sep=None, engine="python", encoding="utf-8-sig", header=None, nrows=2)
        if not preview.empty:
            raw_cols = [str(col).strip() for col in preview.iloc[0].tolist()]
            duplicates = sorted({col for col in raw_cols if raw_cols.count(col) > 1})
            if duplicates:
                raise ValueError("Tệp CSV có cột bị trùng tên: " + ", ".join(duplicates) + ".")
        frame = pd.read_csv(io.BytesIO(data), sep=None, engine="python", encoding="utf-8-sig")
    except ValueError:
        raise
    except Exception as exc:  # pandas raises many parser-specific errors
        raise ValueError(f"Không thể đọc tệp CSV: {exc}") from None
    frame.columns = [str(col).strip() for col in frame.columns]
    if frame.empty:
        raise ValueError("Tệp CSV không có dòng dữ liệu nào.")
    if len(frame) > max_rows:
        raise ValueError(f"Mỗi lần chấm tối đa {max_rows} dòng (tệp hiện có {len(frame)} dòng).")
    missing_required = [col for col in REQUIRED_COLUMNS if col not in frame.columns]
    if missing_required:
        raise ValueError("Tệp CSV thiếu cột bắt buộc: " + ", ".join(missing_required) + ". Hãy dùng tệp mẫu để đối chiếu tên cột.")
    return frame


def score_batch(frame, model_id, artifacts=ARTIFACTS):
    """Validate every row, score valid rows with one verified model load and summarise the batch."""
    registry = read_registry(artifacts)
    path, record = verify_model(registry, model_id, artifacts)
    model = _load_verified(str(path), record["sha256"])
    ignored = list(LAB_FIELDS) if not record.get("uses_labs", True) else []
    ranges = registry.get("training_ranges", {})

    input_cols = [col for col in frame.columns if col in INPUT_FIELDS]
    stripped = [col for col in frame.columns if col in STRIPPED_COLUMNS]
    extra = [col for col in frame.columns if col not in INPUT_FIELDS and col not in STRIPPED_COLUMNS]
    has_reference = TARGET in frame.columns

    rows, valid_frames, valid_positions, missing_by_pos = [], [], [], {}
    for position, (_, series) in enumerate(frame.iterrows()):
        base = {"Dòng": position + 1}
        if ID_COLUMN in frame.columns:
            base["Mã lượt khám"] = _cell(series[ID_COLUMN])
        if has_reference:
            base["Nhãn tham khảo"] = _parse_reference_esi(_cell(series[TARGET]))
        payload = {col: _cell(series[col]) for col in input_cols}
        if payload.get("arrival_timestamp") is not None:
            payload["arrival_timestamp"] = str(payload["arrival_timestamp"])
        if payload.get("site_id") is not None and isinstance(payload["site_id"], str):
            payload["site_id"] = payload["site_id"].strip() or None
        try:
            valid, missing = validate_input(payload)
        except ValueError as exc:
            rows.append({**base, "ok": False, "Lỗi": str(exc)})
            continue
        rows.append({**base, "ok": True, "Lỗi": None})
        valid_frames.append(valid)
        valid_positions.append(position)
        missing_by_pos[position] = [field for field in missing if field not in ignored]

    if valid_frames:
        features = make_features(pd.concat(valid_frames, ignore_index=True))
        probabilities = np.asarray(model.predict_proba(features))
        predicted = np.asarray(model.predict(features)).astype(int)
        if (probabilities.shape != (len(valid_frames), len(CLASSES))
                or not np.isfinite(probabilities).all()
                or np.any(probabilities < 0)
                or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5)):
            raise ValueError("Điểm đầu ra mô hình không hợp lệ.")
        if not set(predicted.tolist()).issubset(CLASSES):
            raise ValueError("Mô hình trả nhãn ESI ngoài phạm vi 1–5.")
        for i, position in enumerate(valid_positions):
            raw = valid_frames[i].iloc[0]
            outside = [field for field, bounds in ranges.items()
                       if field not in ignored and not pd.isna(raw[field])
                       and not bounds["minimum"] <= raw[field] <= bounds["maximum"]]
            result = {"predicted_esi": int(predicted[i]),
                      "scores": {str(c): float(probabilities[i, j]) for j, c in enumerate(CLASSES)},
                      "missing_fields": missing_by_pos[position], "out_of_training_range_fields": outside}
            review = assess_prediction(result, record)
            report = emergency_report(result["predicted_esi"])
            rows[position].update({
                "ESI dự đoán": result["predicted_esi"],
                **{f"ESI {c} (%)": round(result["scores"][str(c)] * 100, 2) for c in CLASSES},
                "Trạng thái": review["label"],
                "Lý do xem xét": " ".join(review["reasons"]),
                "Ưu tiên": report["priority"] if report else "",
            })

    table = pd.DataFrame(rows)
    for int_col in ("ESI dự đoán", "Nhãn tham khảo"):
        if int_col in table.columns:
            table[int_col] = pd.array(table[int_col], dtype="Int64")
    ok = table["ok"] if not table.empty else pd.Series(dtype=bool)
    scored = table[ok] if not table.empty else table
    distribution = {c: int((scored["ESI dự đoán"] == c).sum()) if "ESI dự đoán" in scored.columns else 0 for c in CLASSES}
    summary = {"model_id": model_id, "total_rows": len(table), "succeeded_rows": int(ok.sum()),
               "failed_rows": int((~ok).sum()) if len(ok) else 0,
               "review_required_rows": int((scored["Trạng thái"] == "Chưa đủ cơ sở kết luận").sum()) if "Trạng thái" in scored.columns else 0,
               "esi_distribution": distribution, "stripped_columns": stripped, "ignored_columns": extra,
               "has_reference": has_reference, "reference_matches": None, "reference_rows": 0}
    if has_reference and len(scored) and "ESI dự đoán" in scored.columns:
        labelled = scored[scored["Nhãn tham khảo"].notna()]
        summary["reference_rows"] = int(len(labelled))
        summary["reference_matches"] = int((labelled["Nhãn tham khảo"] == labelled["ESI dự đoán"]).sum())
    return table.drop(columns=["ok"]), summary


def template_csv(registry):
    """CSV template built from the five train examples shipped in the registry."""
    rows = [case["input"] for case in registry["examples"]]
    frame = pd.DataFrame(rows)
    ordered = [col for col in INPUT_FIELDS if col in frame.columns]
    return frame[ordered].to_csv(index=False).encode("utf-8-sig")
