"""Describe urgency for a model prediction without diagnosing or deciding admission."""


def emergency_report(predicted_esi):
    if predicted_esi == 1:
        return {
            "esi_level": 1,
            "priority": "Cấp cứu ngay",
            "recommendation": "Ưu tiên đánh giá và xử trí cấp cứu ngay; bác sĩ xác nhận chỉ định hồi sức hoặc nhập viện.",
        }
    if predicted_esi == 2:
        return {
            "esi_level": 2,
            "priority": "Khẩn cấp",
            "recommendation": "Ưu tiên đánh giá cấp cứu khẩn; bác sĩ xác nhận hướng xử trí và chỉ định nhập viện.",
        }
    return None
