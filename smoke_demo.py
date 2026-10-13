"""Exercise the deployed bundle and frozen models on a clean runtime."""

from pathlib import Path
import math
import sys

ROOT = Path(__file__).resolve().parents[1] if Path(__file__).parent.name == "tools" else Path(__file__).parent
sys.path.insert(0, str(ROOT))

from demo.inference import predict_case, read_registry, validate_input
from fedmml_triage.config import LABS, VITALS
from streamlit.testing.v1 import AppTest


def main():
    registry = read_registry()
    for model_id in registry["models"]:
        for case in registry["examples"]:
            result = predict_case(case["input"], model_id)
            assert result["predicted_esi"] in range(1, 6)
            assert math.isclose(sum(result["scores"].values()), 1, abs_tol=1e-5)
    payload = dict(registry["examples"][0]["input"])
    for age in [0, 17, 101]:
        try:
            validate_input({**payload, "age": age})
        except ValueError:
            pass
        else:
            raise AssertionError("Unsupported age accepted")
    try:
        validate_input({**payload, **{field: "" for field in VITALS}})
    except ValueError:
        pass
    else:
        raise AssertionError("Absent vitals accepted")
    missing_labs = predict_case({**payload, **{field: "" for field in LABS}}, registry["selected_model"])
    assert set(LABS) <= set(missing_labs["missing_fields"])

    at = AppTest.from_file(str(ROOT / "demo" / "app.py"), default_timeout=60).run()
    assert not at.exception
    at.button(key="predict_example").click().run()
    assert not at.exception and not at.error
    assert any(metric.label == "ESI dự đoán" for metric in at.metric)
    assert len(at.session_state["prediction"]["comparisons"]) == len(registry["models"])
    prediction = at.session_state["prediction"]["primary"]["predicted_esi"]
    report = at.session_state["prediction"]["emergency_report"]
    assert (report is not None) == (prediction in [1, 2])
    at.radio(key="input_mode").set_value("Nhập thủ công").run()
    assert at.text_input(key="age").value == ""
    at.button(key="fill_manual_example").click().run()
    at.text_input(key="age").set_value("0")
    at.button(key="predict_manual").click().run()
    assert len(at.error) == 1
    assert not any(metric.label == "ESI dự đoán" for metric in at.metric)
    at.text_input(key="age").set_value("40")
    at.button(key="predict_manual").click().run()
    assert not at.exception and not at.error
    at.text_input(key="heart_rate").set_value("100").run()
    assert "prediction" not in at.session_state
    at.button(key="predict_manual").click().run()
    assert not at.exception and not at.error
    for field in VITALS:
        at.text_input(key=field).set_value("")
    at.button(key="predict_manual").click().run()
    assert len(at.error) == 1
    assert not any(metric.label == "ESI dự đoán" for metric in at.metric)
    at.selectbox(key="model_id").set_value("triage_without_labs").run()
    assert not at.exception
    assert any("Chưa có test độc lập" in item.value for item in at.caption)
    print("PASS: all models, valid examples, input guards, stale-result clearing and validation-only lab-free UI.")


if __name__ == "__main__":
    main()
