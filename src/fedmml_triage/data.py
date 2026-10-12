"""Dataset validation, deterministic sanitation, patient splitting and feature contract."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from .config import BOUNDS, CATEGORICAL, CLASSES, CONTINUOUS, FEATURES, FORBIDDEN, PROCESSED, REPORTS, SEED, SOURCE, SPLIT_PLAN, TARGET
from .io import read_json, sha256, write_json


def load_source(path=SOURCE):
    df = pd.read_csv(path, dtype={"patient_id": str, "encounter_id": str})
    required = ["patient_id", "encounter_id", "arrival_timestamp", TARGET] + CONTINUOUS + CATEGORICAL
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required FedMML columns: {missing}")
    if df[["patient_id", "encounter_id"]].isna().any().any():
        raise ValueError("Patient and encounter identifiers must not be missing.")
    labels = pd.to_numeric(df[TARGET], errors="coerce")
    if not labels.isin(CLASSES).all():
        raise ValueError("esi_level must be an integer in 1..5.")
    df[TARGET] = labels.astype(int)
    return df


def clean_data(df):
    clean = df.copy()
    duplicate_rows = int(clean.duplicated().sum())
    clean = clean.drop_duplicates().reset_index(drop=True)
    if clean.encounter_id.duplicated().any():
        raise ValueError("Conflicting records share an encounter_id; resolve them before training.")
    for col in CATEGORICAL:
        values = clean[col].astype("string").str.strip()
        clean[col] = values.replace("", pd.NA).astype(object).where(values.notna() & values.ne(""), np.nan)
    invalid = {}
    for col in CONTINUOUS:
        values = pd.to_numeric(clean[col], errors="coerce").astype(float)
        low, high = BOUNDS[col]
        bad = values.notna() & (~np.isfinite(values) | values.lt(low) | values.gt(high))
        invalid[col] = int(bad.sum())
        clean[col] = values.mask(bad, np.nan)
    timestamps = pd.to_datetime(clean.arrival_timestamp, errors="coerce")
    if timestamps.dt.tz is not None:
        raise ValueError("arrival_timestamp must be facility-local, timezone-naive.")
    clean.arrival_timestamp = timestamps
    report = {
        "raw_rows": len(df), "clean_rows": len(clean), "exact_duplicates_removed": duplicate_rows,
        "patients": int(clean.patient_id.nunique()), "invalid_numeric_values_to_null": invalid,
        "total_invalid_numeric_values": sum(invalid.values()),
        "invalid_timestamps": int(timestamps.isna().sum()),
        "missing_after_cleaning": {col: int(clean[col].isna().sum()) for col in CONTINUOUS + CATEGORICAL + ["arrival_timestamp"]},
        "missing_value_policy": "retain NaN here; fit imputation inside each training/CV pipeline",
        "outlier_policy": "fixed validity bounds here; 0.5%/99.5% winsorization fitted on training rows only",
    }
    return clean, report


def split_patients(df, plan=None, seed=SEED):
    if plan is not None:
        groups = {name: set(ids) for name, ids in plan["patients"].items()}
        for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
            if groups[a] & groups[b]:
                raise ValueError(f"Patient leakage: {a}/{b}")
        if set.union(*groups.values()) != set(df.patient_id.unique()):
            raise ValueError("Frozen split plan does not cover exactly the source patients.")
        splits = {name: df[df.patient_id.isin(ids)].copy() for name, ids in groups.items()}
    else:
        outer = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
        remaining, test = next(outer.split(df, groups=df.patient_id))
        rest = df.iloc[remaining]
        inner = GroupShuffleSplit(n_splits=1, test_size=0.15 / 0.85, random_state=seed)
        train, validation = next(inner.split(rest, groups=rest.patient_id))
        splits = {"train": rest.iloc[train].copy(), "validation": rest.iloc[validation].copy(), "test": df.iloc[test].copy()}
    assert_no_patient_overlap(splits)
    for name, part in splits.items():
        if set(part[TARGET].unique()) != set(CLASSES):
            raise ValueError(f"{name} lacks one of the five ESI classes.")
    return splits


def assert_no_patient_overlap(splits):
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        if set(splits[a].patient_id) & set(splits[b].patient_id):
            raise ValueError(f"Patient leakage: {a}/{b}")


def make_features(df):
    features = df[CONTINUOUS + CATEGORICAL].copy()
    ts = pd.to_datetime(df.arrival_timestamp, errors="coerce")
    features["arrival_hour"] = ts.dt.hour.astype(float)
    features["arrival_day_of_week"] = ts.dt.dayofweek.astype(float)
    features["arrival_month"] = ts.dt.month.astype(float)
    features["is_weekend"] = (ts.dt.dayofweek >= 5).astype(float).where(ts.notna(), np.nan)
    for col in CATEGORICAL:
        features[col] = features[col].astype(object).where(features[col].notna(), np.nan)
    features = features[FEATURES]
    if set(features.columns) & set(FORBIDDEN):
        raise AssertionError("A forbidden identifier, target or text field entered the model.")
    return features


def prepare_data(source=SOURCE, output=PROCESSED, report_dir=REPORTS, split_plan=SPLIT_PLAN):
    source, output, report_dir = Path(source), Path(output), Path(report_dir)
    raw = load_source(source)
    clean, cleaning = clean_data(raw)
    plan = read_json(split_plan) if split_plan and Path(split_plan).is_file() else None
    source_hash = sha256(source)
    if plan and plan["source_sha256"] != source_hash:
        raise ValueError("Source checksum differs from the frozen split plan.")
    splits = split_patients(clean, plan=plan)
    report = export_prepared(raw, clean, splits, cleaning, source, output, report_dir, bool(plan and plan.get("historical_test_exposure")))
    return raw, clean, splits, report


def export_prepared(raw, clean, splits, cleaning, source=SOURCE, output=PROCESSED, report_dir=REPORTS, historical_test_exposure=True):
    source, output, report_dir = Path(source), Path(output), Path(report_dir)
    source_hash = sha256(source)
    assert_no_patient_overlap(splits)
    output.mkdir(parents=True, exist_ok=True)
    split_report = {}
    for name, frame in splits.items():
        destination = output / f"{name}.csv"
        frame.to_csv(destination, index=False)
        split_report[name] = {
            "rows": len(frame), "patients": int(frame.patient_id.nunique()),
            "class_counts": {str(c): int((frame[TARGET] == c).sum()) for c in CLASSES},
            "sha256": sha256(destination), "file": str(destination.relative_to(output.parent.parent)),
        }
    report = {
        "dataset": "FedMML", "source_sha256": source_hash, "cleaning": cleaning,
        "splits": split_report, "patient_overlap": 0, "seed": SEED,
        "historical_test_exposure": historical_test_exposure,
        "forbidden_model_inputs": FORBIDDEN, "model_features": FEATURES,
    }
    write_json(report_dir / "preprocessing.json", report)
    return report
