"""
Fraud Detection API
====================
Serves the trained sklearn Pipeline (UID feature engineering -> imputation ->
encoding -> feature selection -> XGBoost classifier) behind a FastAPI
/predict endpoint.

Run locally:
    uvicorn app.main:app --host 0.0.0.0 --port 8000

Docs (Swagger UI):
    http://localhost:8000/docs
"""
import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve,
)

from features.feature_construction import transaction_amt, TransactionAmt_decimal, uid

warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR.parent / "model"

# ---------------------------------------------------------------------------
# Load model, threshold, and schema at startup
# ---------------------------------------------------------------------------
model = joblib.load(MODEL_DIR / "model.pkl")
threshold = float(joblib.load(MODEL_DIR / "threshold.pkl"))

with open(BASE_DIR / "schema.json") as f:
    SCHEMA = json.load(f)

ALL_COLUMNS: List[str] = SCHEMA["all_columns"]           # 426 raw columns the pipeline needs (pre uid_features step)
CATEGORICAL_COLUMNS: List[str] = SCHEMA["categorical_columns"]
NUMERIC_COLUMNS: List[str] = SCHEMA["numeric_columns"]

# Columns required to build engineered features (uid, log, decimal amount).
# Everything else can be missing and will be imputed by the pipeline.
REQUIRED_FIELDS = ["TransactionAmt", "card1", "card2", "card3", "card5", "addr1", "addr2"]

app = FastAPI(
    title="Fraud Detection API",
    description="Scores a transaction for fraud probability using the trained XGBoost pipeline.",
    version="1.0.0",
)

# Allows the local dashboard/checkout HTML pages (opened as file:// pages)
# to call this API from the browser. Tighten allow_origins before deploying
# anywhere public.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Load the real held-out evaluation set (x_test / y_test carved out of the
# labeled IEEE-CIS training data during your original main.py run) and score
# it once at startup. Every dashboard/performance/investigation endpoint
# below reads from this in-memory table -- nothing here is invented.
# ---------------------------------------------------------------------------
DATA_DIR = BASE_DIR.parent / "data"
_x_test_path = DATA_DIR / "x_test_sample.csv.gz"
_y_test_path = DATA_DIR / "y_test_sample.csv"

EVAL_AVAILABLE = _x_test_path.exists() and _y_test_path.exists()
EVAL_DF: Optional[pd.DataFrame] = None

if EVAL_AVAILABLE:
    _x_test = pd.read_csv(_x_test_path)
    _y_test = pd.read_csv(_y_test_path)["isFraud"]

    _proba = model.predict_proba(_x_test)[:, 1]
    _pred = (_proba >= threshold).astype(int)

    EVAL_DF = _x_test.copy()
    EVAL_DF["isFraud"] = _y_test.values
    EVAL_DF["fraud_probability"] = _proba
    EVAL_DF["predicted"] = _pred
    EVAL_DF["correct"] = EVAL_DF["predicted"] == EVAL_DF["isFraud"]

    def _status_row(r):
        if r["isFraud"] == 1 and r["predicted"] == 1:
            return "true_positive"
        if r["isFraud"] == 0 and r["predicted"] == 0:
            return "true_negative"
        if r["isFraud"] == 0 and r["predicted"] == 1:
            return "false_positive"
        return "false_negative"

    EVAL_DF["status"] = EVAL_DF.apply(_status_row, axis=1)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------
class Transaction(BaseModel):
    """
    A single transaction record. Only the fields below are strictly required
    (they're needed to build the uid / log-amount / decimal-amount features).
    Any other field from the original schema (V1-V339, C1-C14, D1-D15, M1-M9,
    id_01-id_38, dist1, dist2, ProductCD, card4, card6, DeviceType,
    DeviceInfo, TransactionID, TransactionDT, ...) may be included too -- pass
    them inside `extra_fields`. Omitted fields are treated as missing/NaN and
    handled by the pipeline's built-in imputers.
    """

    TransactionAmt: float = Field(..., description="Transaction amount")
    card1: float = Field(..., description="Card identifier 1")
    card2: Optional[float] = Field(None, description="Card identifier 2")
    card3: Optional[float] = Field(None, description="Card identifier 3")
    card5: Optional[float] = Field(None, description="Card identifier 5")
    addr1: Optional[float] = Field(None, description="Billing address code 1")
    addr2: Optional[float] = Field(None, description="Billing address code 2")

    extra_fields: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Any additional raw columns from the original schema "
            "(e.g. V1, D4, C13, id_02, ProductCD, card4, DeviceType, ...). "
            "See GET /schema for the full list of accepted column names."
        ),
    )


class PredictionResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool
    threshold: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def build_dataframe(txn: Transaction) -> pd.DataFrame:
    row: Dict[str, Any] = {col: np.nan for col in ALL_COLUMNS if col != "uid"}

    # Core typed fields
    row["TransactionAmt"] = txn.TransactionAmt
    row["card1"] = txn.card1
    row["card2"] = txn.card2
    row["card3"] = txn.card3
    row["card5"] = txn.card5
    row["addr1"] = txn.addr1
    row["addr2"] = txn.addr2

    # Any extra raw columns the caller supplied
    unknown = []
    for key, value in txn.extra_fields.items():
        if key in row:
            row[key] = value
        else:
            unknown.append(key)
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown field(s) not part of the model schema: {unknown}. "
                   f"See GET /schema for accepted column names.",
        )

    df = pd.DataFrame([row])

    # Required before the pipeline can build the uid-based features
    if df["card1"].isna().any():
        raise HTTPException(status_code=422, detail="card1 is required to build the uid feature.")

    df = transaction_amt(df)          # adds TransactionAmt_log
    df = TransactionAmt_decimal(df)   # adds TransactionAmt_decimal
    df = uid(df)                      # adds uid (card1_card2_card3_card5_addr1_addr2)

    # Reorder to match what the pipeline's first step expects (order doesn't
    # strictly matter for sklearn ColumnTransformer since it selects by name,
    # but keeping consistent ordering avoids surprises).
    ordered_cols = ALL_COLUMNS + ["uid"] if "uid" not in ALL_COLUMNS else ALL_COLUMNS
    df = df[[c for c in ordered_cols if c in df.columns]]

    return df


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "threshold": threshold}


@app.get("/schema")
def schema():
    """Full list of raw column names the model accepts, split by type."""
    return {
        "required_fields": REQUIRED_FIELDS,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "numeric_columns": NUMERIC_COLUMNS,
        "total_columns": len(ALL_COLUMNS),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(txn: Transaction):
    df = build_dataframe(txn)
    try:
        proba = model.predict_proba(df)[:, 1][0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {e}")

    return PredictionResponse(
        fraud_probability=float(proba),
        is_fraud=bool(proba >= threshold),
        threshold=threshold,
    )


# ---------------------------------------------------------------------------
# Dashboard / Transaction Investigation / Model Performance
# All of these read from EVAL_DF -- the real held-out test set scored at
# startup. If data/x_test_sample.csv + y_test_sample.csv aren't present,
# these endpoints return a clear 503 rather than fabricating numbers.
# ---------------------------------------------------------------------------
def _require_eval_data():
    if not EVAL_AVAILABLE or EVAL_DF is None:
        raise HTTPException(
            status_code=503,
            detail="No held-out evaluation data found at data/x_test_sample.csv "
                   "and data/y_test_sample.csv. Dashboard/Performance/Investigation "
                   "endpoints need real labeled test data to avoid showing fake numbers.",
        )


def _human_row(r: pd.Series) -> Dict[str, Any]:
    """Human-readable view of one transaction. Anonymized ML features are
    labeled honestly as anonymized -- never renamed into something they're
    not (e.g. card1 is never called 'Card Number')."""
    return {
        "transaction_id": int(r["TransactionID"]) if not pd.isna(r.get("TransactionID")) else None,
        "amount": None if pd.isna(r.get("TransactionAmt")) else round(float(r["TransactionAmt"]), 2),
        "product_category": None if pd.isna(r.get("ProductCD")) else str(r["ProductCD"]),
        "card_network": None if pd.isna(r.get("card4")) else str(r["card4"]),
        "payment_method": None if pd.isna(r.get("card6")) else str(r["card6"]),
        "device_type": None if pd.isna(r.get("DeviceType")) else str(r["DeviceType"]),
        "address_region_code": {
            "addr1": None if pd.isna(r.get("addr1")) else float(r["addr1"]),
            "addr2": None if pd.isna(r.get("addr2")) else float(r["addr2"]),
            "note": "Anonymized regional code from the dataset, not a real address.",
        },
        "card_feature_anonymized": {
            "card1": None if pd.isna(r.get("card1")) else float(r["card1"]),
            "card2": None if pd.isna(r.get("card2")) else float(r["card2"]),
            "card3": None if pd.isna(r.get("card3")) else float(r["card3"]),
            "card5": None if pd.isna(r.get("card5")) else float(r["card5"]),
            "note": "Anonymized card identifiers from the dataset, not a real card number.",
        },
        "actual_label": "fraud" if r["isFraud"] == 1 else "legitimate",
        "predicted_label": "fraud" if r["predicted"] == 1 else "legitimate",
        "fraud_probability": round(float(r["fraud_probability"]), 4),
        "status": r["status"],
        "correct": bool(r["correct"]),
    }


@app.get("/dashboard/summary")
def dashboard_summary():
    """KPI cards + confusion matrix + actual-vs-predicted + class distribution.
    Source: held-out test split of the labeled IEEE-CIS training data."""
    _require_eval_data()
    y_true = EVAL_DF["isFraud"]
    y_pred = EVAL_DF["predicted"]
    y_proba = EVAL_DF["fraud_probability"]

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "dataset_source": "Held-out split of labeled IEEE-CIS training data (NOT the official unlabeled Kaggle test set)",
        "model": "XGBoost",
        "threshold": threshold,
        "kpis": {
            "total_transactions": int(len(EVAL_DF)),
            "actual_fraud": int(y_true.sum()),
            "predicted_fraud": int(y_pred.sum()),
            "correct_predictions": int(EVAL_DF["correct"].sum()),
            "fraud_rate": round(float(y_true.mean()) * 100, 3),
            "precision": round(precision_score(y_true, y_pred), 4),
            "recall": round(recall_score(y_true, y_pred), 4),
            "f1_score": round(f1_score(y_true, y_pred), 4),
            "roc_auc": round(roc_auc_score(y_true, y_proba), 4),
            "pr_auc": round(average_precision_score(y_true, y_proba), 4),
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
        },
        "confusion_matrix": {
            "true_negative": int(tn), "false_positive": int(fp),
            "false_negative": int(fn), "true_positive": int(tp),
        },
        "actual_vs_predicted": {
            "actual_legitimate": int((y_true == 0).sum()),
            "predicted_legitimate": int((y_pred == 0).sum()),
            "actual_fraud": int((y_true == 1).sum()),
            "predicted_fraud": int((y_pred == 1).sum()),
        },
        "class_distribution": {
            "legitimate": int((y_true == 0).sum()),
            "fraud": int((y_true == 1).sum()),
        },
    }


@app.get("/dashboard/recent")
def dashboard_recent(limit: int = Query(20, ge=1, le=200)):
    """Recent transactions table for the dashboard."""
    _require_eval_data()
    rows = EVAL_DF.head(limit)
    return [_human_row(r) for _, r in rows.iterrows()]


@app.get("/transactions")
def list_transactions(
    search: Optional[str] = Query(None, description="Search by Transaction ID"),
    filter: str = Query("all", pattern="^(all|fraud|legitimate|correct|incorrect)$"),
    limit: int = Query(50, ge=1, le=500),
):
    """List/search test transactions for the Transaction Investigation page."""
    _require_eval_data()
    df = EVAL_DF

    if search:
        try:
            search_id = int(search)
            df = df[df["TransactionID"] == search_id]
        except ValueError:
            df = df.iloc[0:0]

    if filter == "fraud":
        df = df[df["isFraud"] == 1]
    elif filter == "legitimate":
        df = df[df["isFraud"] == 0]
    elif filter == "correct":
        df = df[df["correct"]]
    elif filter == "incorrect":
        df = df[~df["correct"]]

    df = df.head(limit)
    return [_human_row(r) for _, r in df.iterrows()]


@app.get("/transactions/{transaction_id}")
def get_transaction(transaction_id: int):
    """Full human-readable detail for one transaction, auto-fetched from the
    held-out test set -- the user never types raw ML feature values."""
    _require_eval_data()
    match = EVAL_DF[EVAL_DF["TransactionID"] == transaction_id]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"Transaction {transaction_id} not found in the held-out set.")
    return _human_row(match.iloc[0])


@app.post("/transactions/{transaction_id}/analyze")
def analyze_transaction(transaction_id: int):
    """Re-runs the full row through the real pipeline live (rather than just
    reading the cached prediction) and compares to the actual label."""
    _require_eval_data()
    match = EVAL_DF[EVAL_DF["TransactionID"] == transaction_id]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"Transaction {transaction_id} not found in the held-out set.")

    row = match.iloc[0]
    feature_cols = [c for c in ALL_COLUMNS if c != "uid"] + ["uid"]
    x = match[feature_cols]

    proba = float(model.predict_proba(x)[:, 1][0])
    is_fraud = proba >= threshold
    actual_fraud = bool(row["isFraud"] == 1)

    if actual_fraud and is_fraud:
        status = "true_positive"
    elif (not actual_fraud) and (not is_fraud):
        status = "true_negative"
    elif (not actual_fraud) and is_fraud:
        status = "false_positive"
    else:
        status = "false_negative"

    return {
        "transaction_id": transaction_id,
        "fraud_probability": round(proba, 4),
        "threshold": threshold,
        "model_prediction": "fraud" if is_fraud else "legitimate",
        "actual_label": "fraud" if actual_fraud else "legitimate",
        "status": status,
        "correct": status in ("true_positive", "true_negative"),
    }


@app.get("/performance")
def performance():
    """Model Performance page: metrics, ROC curve, PR curve, confusion
    matrix, model/dataset info -- all computed from the real held-out set."""
    _require_eval_data()
    y_true = EVAL_DF["isFraud"]
    y_pred = EVAL_DF["predicted"]
    y_proba = EVAL_DF["fraud_probability"]

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    prec, rec, _ = precision_recall_curve(y_true, y_proba)

    # Downsample curve points for a lighter payload -- ~100 points is plenty for a chart
    def _downsample(a, b, n=100):
        if len(a) <= n:
            return list(a), list(b)
        idx = np.linspace(0, len(a) - 1, n).astype(int)
        return list(np.array(a)[idx]), list(np.array(b)[idx])

    fpr_s, tpr_s = _downsample(fpr, tpr)
    rec_s, prec_s = _downsample(rec, prec)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "model_info": {
            "model": "XGBoost",
            "dataset": "IEEE-CIS (held-out split of labeled training data)",
            "evaluation": "Held-out test split",
            "threshold": threshold,
            "num_test_transactions": int(len(EVAL_DF)),
            "fraud_percentage": round(float(y_true.mean()) * 100, 3),
        },
        "metrics": {
            "precision": round(precision_score(y_true, y_pred), 4),
            "recall": round(recall_score(y_true, y_pred), 4),
            "f1_score": round(f1_score(y_true, y_pred), 4),
            "roc_auc": round(roc_auc_score(y_true, y_proba), 4),
            "pr_auc": round(average_precision_score(y_true, y_proba), 4),
        },
        "confusion_matrix": {
            "true_negative": int(tn), "false_positive": int(fp),
            "false_negative": int(fn), "true_positive": int(tp),
        },
        "roc_curve": {"fpr": [round(float(x), 4) for x in fpr_s], "tpr": [round(float(x), 4) for x in tpr_s]},
        "pr_curve": {"recall": [round(float(x), 4) for x in rec_s], "precision": [round(float(x), 4) for x in prec_s]},
    }
