from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data" / "raw" / "fedmml_ed_triage_dataset.csv"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "outputs" / "reports"
EXPERIMENTS = ROOT / "outputs" / "experiments"
SPLIT_PLAN = ROOT / "configs" / "split_plan.json"
SEED = 42
CLASSES = (1, 2, 3, 4, 5)
TARGET = "esi_level"
VITALS = ["systolic_bp", "diastolic_bp", "heart_rate", "respiratory_rate", "temperature", "spo2", "pain_score"]
LABS = ["wbc", "hemoglobin", "platelet_count", "sodium", "potassium", "creatinine", "glucose", "troponin", "bnp", "lactate", "inr"]
CONTINUOUS = ["age"] + VITALS + LABS
TEMPORAL = ["arrival_hour", "arrival_day_of_week", "arrival_month", "is_weekend"]
NUMERIC = CONTINUOUS + TEMPORAL
CATEGORICAL = ["sex", "site_id", "country"]
FEATURES = NUMERIC + CATEGORICAL
FORBIDDEN = ["patient_id", "encounter_id", TARGET, "chief_complaint", "clinical_notes", "arrival_timestamp"]

# Engineering validity rules inherited from the dataset workflow, not certified clinical cutoffs.
# Valid extremes remain available; train-fitted winsorization is a separate preprocessing step.
BOUNDS = {
    "age": (0, 125), "systolic_bp": (40, 300), "diastolic_bp": (20, 200),
    "heart_rate": (20, 300), "respiratory_rate": (4, 80), "temperature": (25, 45),
    "spo2": (0, 100), "pain_score": (0, 10), "wbc": (0, 200),
    "hemoglobin": (0, 30), "platelet_count": (0, 2000), "sodium": (50, 200),
    "potassium": (1, 15), "creatinine": (0, 30), "glucose": (0, 1500),
    "troponin": (0, 100), "bnp": (0, 10000), "lactate": (0, 35), "inr": (0, 20),
}

PROTOCOL = {
    "dataset": "FedMML Emergency Department Triage Dataset",
    "target": TARGET,
    "classes": list(CLASSES),
    "seed": SEED,
    "cv_folds": 3,
    "search_iterations_per_algorithm": 6,
    "primary_cv_metric": "macro_f1",
    "winner_selection": "validation macro_f1, then macro_recall, then macro ROC-AUC OvR",
    "test_usage": "compare the three frozen models once; reuse verified results on subsequent runs",
    "historical_test_exposure": True,
    "features": FEATURES,
    "outlier_quantiles": [0.005, 0.995],
    "research_only": True,
}
