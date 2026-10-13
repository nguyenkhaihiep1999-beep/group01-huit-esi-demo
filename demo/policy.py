"""Research input completeness and validation-derived review indicators."""

CORE_VITALS = ("systolic_bp", "diastolic_bp", "heart_rate", "respiratory_rate", "temperature", "spo2")
VITAL_NAMES = {
    "systolic_bp": "huyết áp tâm thu", "diastolic_bp": "huyết áp tâm trương",
    "heart_rate": "nhịp tim", "respiratory_rate": "nhịp thở",
    "temperature": "nhiệt độ", "spo2": "SpO₂",
}
LAB_FIELDS = ("wbc", "hemoglobin", "platelet_count", "sodium", "potassium", "creatinine",
              "glucose", "troponin", "bnp", "lactate", "inr")


def assess_prediction(result, model_record):
    ranked = sorted(result["scores"].items(), key=lambda pair: (-pair[1], int(pair[0])))
    gap = float(ranked[0][1] - ranked[1][1])
    policy = model_record.get("review_policy")
    threshold = policy["minimum_margin"] if policy else None
    missing = set(result["missing_fields"])
    reasons = []
    outside = result.get("out_of_training_range_fields", [])
    if outside:
        reasons.append("Có chỉ số nằm ngoài khoảng quan sát trong tập train: " + ", ".join(VITAL_NAMES.get(field, field) for field in outside) + ".")
    if model_record.get("uses_labs", True) and missing.intersection(LAB_FIELDS):
        reasons.append("Mô hình dùng xét nghiệm nhưng đầu vào thiếu kết quả; dự đoán phụ thuộc giá trị điền.")
    if missing.intersection({"sex", "country", "site_id"}):
        reasons.append("Thông tin giới tính, quốc gia hoặc cơ sở chưa đầy đủ so với đầu vào đã huấn luyện.")
    if threshold is None:
        reasons.append("Chưa có ngưỡng xem xét được đánh giá trên validation cho mô hình này.")
    elif gap < threshold:
        reasons.append(f"Hai mức đứng đầu gần nhau: ESI {ranked[0][0]} và ESI {ranked[1][0]} chênh {gap * 100:.2f} điểm phần trăm.")
    return {
        "status": "review_required" if reasons else "reference_only",
        "label": "Chưa đủ cơ sở kết luận" if reasons else "Kết quả tham khảo",
        "reasons": reasons,
        "top_two": [{"esi_level": int(level), "score": float(score)} for level, score in ranked[:2]],
        "score_margin": gap, "minimum_margin": threshold, "threshold_selected_on": "validation" if policy else None,
        "probability_calibrated": False, "clinical_cutoff": False,
    }
