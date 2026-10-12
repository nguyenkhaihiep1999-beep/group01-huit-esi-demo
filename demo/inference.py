"""Validate raw demo inputs and load only checksum-verified frozen pipelines."""

from functools import lru_cache
from importlib.metadata import version
import json
import math
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fedmml_triage.config import BOUNDS, CATEGORICAL, CLASSES, CONTINUOUS, VITALS
from fedmml_triage.data import make_features
from fedmml_triage.io import sha256

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
INPUT_FIELDS = CONTINUOUS + CATEGORICAL + ["arrival_timestamp"]


def read_registry(artifacts=ARTIFACTS):
    artifacts = Path(artifacts)
    path = artifacts / "model_registry.json"
    expected = (artifacts / "model_registry.sha256").read_text(encoding="utf-8").strip()
    if sha256(path) != expected:
        raise ValueError("Bộ thông tin mô hình bị thay đổi; hãy tạo lại gói demo từ artifact đã xác minh.")
    registry = json.loads(path.read_text(encoding="utf-8"))
    if registry["classes"] != list(CLASSES):
        raise ValueError("Thứ tự nhãn ESI không hợp lệ.")
    return registry


def validate_input(payload):
    unknown = set(payload) - set(INPUT_FIELDS)
    if unknown:
        raise ValueError("Trường không được dùng làm đầu vào: " + ", ".join(sorted(unknown)))
    row = {}
    for field in CONTINUOUS:
        value = payload.get(field)
        if value is None or (isinstance(value, str) and not value.strip()):
            row[field] = np.nan
            continue
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{field}: cần một số hoặc để trống.")
        try:
            number = float(value.strip().replace(",", ".") if isinstance(value, str) else value)
        except (TypeError, ValueError):
            raise ValueError(f"{field}: cần một số hoặc để trống.") from None
        low, high = BOUNDS[field]
        if not math.isfinite(number) or not low <= number <= high:
            raise ValueError(f"{field}: giá trị hợp lệ của schema nằm trong [{low}, {high}].")
        row[field] = number
    if pd.isna(row["age"]):
        raise ValueError("Vui lòng nhập tuổi.")
    if not 18 <= row["age"] <= 100:
        raise ValueError("age: demo chỉ hỗ trợ tuổi 18–100, theo phạm vi dữ liệu huấn luyện tổng hợp.")
    if all(pd.isna(row[field]) for field in VITALS):
        raise ValueError("Vui lòng nhập ít nhất một dấu hiệu sinh tồn; không dự đoán khi toàn bộ sinh hiệu để trống.")
    for field in ["sex", "country"]:
        value = payload.get(field)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"{field}: cần một giá trị văn bản.")
        row[field] = value.strip() if isinstance(value, str) and value.strip() else np.nan
    site = payload.get("site_id")
    if site is None or (isinstance(site, str) and not site.strip()):
        row["site_id"] = np.nan
    else:
        try:
            site_number = float(site)
            if isinstance(site, (bool, np.bool_)) or not math.isfinite(site_number) or site_number < 1 or not site_number.is_integer():
                raise ValueError
            row["site_id"] = int(site_number)
        except (TypeError, ValueError):
            raise ValueError("site_id: cần mã cơ sở là số nguyên dương.") from None
    try:
        timestamp = pd.Timestamp(payload.get("arrival_timestamp"))
        if pd.isna(timestamp) or timestamp.tzinfo is not None:
            raise ValueError
    except (TypeError, ValueError):
        raise ValueError("Thời điểm tiếp nhận phải là ngày/giờ địa phương, không kèm múi giờ.") from None
    row["arrival_timestamp"] = timestamp
    frame = pd.DataFrame([row])
    missing = [field for field in CONTINUOUS + CATEGORICAL if pd.isna(row[field])]
    return frame, missing


def verify_model(registry, model_id, artifacts=ARTIFACTS):
    if model_id not in registry["models"]:
        raise ValueError("Mô hình không có trong gói demo.")
    for package, expected in registry["runtime_versions"].items():
        if version(package) != expected:
            raise ValueError(f"Phiên bản {package} cần là {expected}; hãy cài demo/requirements.txt.")
    root = Path(artifacts).resolve()
    record = registry["models"][model_id]
    path = (root / record["file"]).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Đường dẫn mô hình nằm ngoài gói demo.")
    if sha256(path) != record["sha256"]:
        raise ValueError("Tệp mô hình không khớp SHA-256; dừng dự đoán.")
    return path, record


@lru_cache(maxsize=4)
def _load_verified(path, expected_hash):
    # Verify again immediately before deserialization. No uploaded pickle files are accepted.
    if sha256(path) != expected_hash:
        raise ValueError("Tệp mô hình bị thay đổi trước khi nạp.")
    model = joblib.load(path)
    if not np.array_equal(model.classes_, CLASSES):
        raise ValueError("Pipeline trả sai thứ tự nhãn ESI.")
    return model


def predict_case(payload, model_id, artifacts=ARTIFACTS):
    frame, missing = validate_input(payload)
    registry = read_registry(artifacts)
    path, record = verify_model(registry, model_id, artifacts)
    model = _load_verified(str(path), record["sha256"])
    probabilities = np.asarray(model.predict_proba(make_features(frame)))[0]
    if probabilities.shape != (5,) or not np.isfinite(probabilities).all() or np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1, atol=1e-5):
        raise ValueError("Điểm đầu ra mô hình không hợp lệ.")
    predicted = int(model.predict(make_features(frame))[0])
    if predicted not in CLASSES:
        raise ValueError("Mô hình trả nhãn ESI ngoài phạm vi 1–5.")
    return {"model_id": model_id, "model_label": record["label"], "predicted_esi": predicted,
            "scores": {str(c): float(probabilities[i]) for i, c in enumerate(CLASSES)},
            "missing_fields": missing, "probability_calibrated": False}
