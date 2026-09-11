"""
Fraud Detection API
====================
Serves the trained sklearn Pipeline (UID feature engineering -> imputation ->
encoding -> feature selection -> XGBoost classifier) behind a FastAPI
/predict endpoint.

The held-out evaluation dataset contains 100,000 transactions. To stay within
low-memory hosting limits, the evaluation CSV is processed in batches instead
of loading the complete 427-column dataset into RAM.

Run locally:
    uvicorn app.main:app --host 0.0.0.0 --port 8000

Docs:
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
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)

from features.feature_construction import (
    transaction_amt,
    TransactionAmt_decimal,
    uid,
)


warnings.filterwarnings("ignore", category=UserWarning)


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR.parent / "model"
DATA_DIR = BASE_DIR.parent / "data"


# ---------------------------------------------------------------------------
# Load model, threshold, and schema at startup
# ---------------------------------------------------------------------------

model = joblib.load(MODEL_DIR / "model.pkl")
threshold = float(joblib.load(MODEL_DIR / "threshold.pkl"))

with open(BASE_DIR / "schema.json") as f:
    SCHEMA = json.load(f)


ALL_COLUMNS: List[str] = SCHEMA["all_columns"]
CATEGORICAL_COLUMNS: List[str] = SCHEMA["categorical_columns"]
NUMERIC_COLUMNS: List[str] = SCHEMA["numeric_columns"]


# Columns required to build engineered features.
REQUIRED_FIELDS = [
    "TransactionAmt",
    "card1",
    "card2",
    "card3",
    "card5",
    "addr1",
    "addr2",
]


# ---------------------------------------------------------------------------
# Evaluation dataset
# ---------------------------------------------------------------------------

_x_test_path = DATA_DIR / "x_test_sample.csv.gz"
_y_test_path = DATA_DIR / "y_test_sample.csv"

EVAL_AVAILABLE = _x_test_path.exists() and _y_test_path.exists()

# Number of rows processed at once.
# 2,000 keeps temporary pandas/model memory relatively small.
EVAL_CHUNK_SIZE = 2000


# ---------------------------------------------------------------------------
# Lightweight evaluation result
#
# IMPORTANT:
# We DO NOT keep all 427 raw feature columns here.
#
# Instead, this table contains only the information needed by the dashboard,
# performance page, and transaction filtering.
# ---------------------------------------------------------------------------

EVAL_DF: Optional[pd.DataFrame] = None


def _status_from_arrays(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> np.ndarray:
    """
    Create transaction status labels without using DataFrame.apply().
    This is much lighter on memory.
    """

    status = np.empty(len(actual), dtype=object)

    status[(actual == 1) & (predicted == 1)] = "true_positive"
    status[(actual == 0) & (predicted == 0)] = "true_negative"
    status[(actual == 0) & (predicted == 1)] = "false_positive"
    status[(actual == 1) & (predicted == 0)] = "false_negative"

    return status


def _load_evaluation_results():
    """
    Process the complete 100,000-row held-out dataset in small batches.

    The raw 427-column data is never kept entirely in RAM.

    Only these lightweight columns are retained:

        TransactionID
        isFraud
        fraud_probability
        predicted
        correct
        status
    """

    if not EVAL_AVAILABLE:
        return None

    # Read labels. This file is tiny compared with x_test.
    y_test = pd.read_csv(_y_test_path)["isFraud"].to_numpy()

    # We don't know the exact number of rows from the compressed file without
    # reading it, so collect small NumPy arrays and combine them afterwards.
    transaction_ids = []
    actual_labels = []
    probabilities = []
    predictions = []

    processed = 0

    for chunk in pd.read_csv(
        _x_test_path,
        compression="gzip",
        chunksize=EVAL_CHUNK_SIZE,
    ):
        # Make sure the corresponding labels are aligned with this chunk.
        chunk_size = len(chunk)

        y_chunk = y_test[processed:processed + chunk_size]

        if len(y_chunk) != chunk_size:
            raise RuntimeError(
                "x_test and y_test row counts do not match."
            )

        # Run the real trained pipeline.
        proba_chunk = model.predict_proba(chunk)[:, 1]

        pred_chunk = (proba_chunk >= threshold).astype(np.int8)

        # Keep ONLY lightweight information.
        transaction_ids.append(
            chunk["TransactionID"].to_numpy()
        )

        actual_labels.append(
            y_chunk.astype(np.int8)
        )

        probabilities.append(
            proba_chunk.astype(np.float32)
        )

        predictions.append(pred_chunk)

        processed += chunk_size

        # The chunk goes out of scope at the next iteration.
        del chunk
        del proba_chunk
        del pred_chunk
        del y_chunk

    # Combine the small arrays.
    transaction_ids = np.concatenate(transaction_ids)
    actual_labels = np.concatenate(actual_labels)
    probabilities = np.concatenate(probabilities)
    predictions = np.concatenate(predictions)

    correct = predictions == actual_labels

    status = _status_from_arrays(
        actual_labels,
        predictions,
    )

    # This DataFrame is small because it contains only 6 columns.
    result = pd.DataFrame(
        {
            "TransactionID": transaction_ids,
            "isFraud": actual_labels,
            "fraud_probability": probabilities,
            "predicted": predictions,
            "correct": correct,
            "status": status,
        }
    )

    return result


if EVAL_AVAILABLE:
    EVAL_DF = _load_evaluation_results()


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Fraud Detection API",
    description=(
        "Scores transactions for fraud probability using the trained "
        "XGBoost pipeline."
    ),
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class Transaction(BaseModel):
    """
    A single transaction record.

    Only the fields below are strictly required.
    Additional raw model fields can be supplied through extra_fields.
    """

    TransactionAmt: float = Field(
        ...,
        description="Transaction amount",
    )

    card1: float = Field(
        ...,
        description="Card identifier 1",
    )

    card2: Optional[float] = Field(
        None,
        description="Card identifier 2",
    )

    card3: Optional[float] = Field(
        None,
        description="Card identifier 3",
    )

    card5: Optional[float] = Field(
        None,
        description="Card identifier 5",
    )

    addr1: Optional[float] = Field(
        None,
        description="Billing address code 1",
    )

    addr2: Optional[float] = Field(
        None,
        description="Billing address code 2",
    )

    extra_fields: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Additional raw columns from the original schema."
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

    row: Dict[str, Any] = {
        col: np.nan
        for col in ALL_COLUMNS
        if col != "uid"
    }

    row["TransactionAmt"] = txn.TransactionAmt
    row["card1"] = txn.card1
    row["card2"] = txn.card2
    row["card3"] = txn.card3
    row["card5"] = txn.card5
    row["addr1"] = txn.addr1
    row["addr2"] = txn.addr2

    unknown = []

    for key, value in txn.extra_fields.items():

        if key in row:
            row[key] = value
        else:
            unknown.append(key)

    if unknown:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unknown field(s) not part of the model schema: {unknown}. "
                f"See GET /schema for accepted column names."
            ),
        )

    df = pd.DataFrame([row])

    if df["card1"].isna().any():
        raise HTTPException(
            status_code=422,
            detail="card1 is required to build the uid feature.",
        )

    df = transaction_amt(df)
    df = TransactionAmt_decimal(df)
    df = uid(df)

    ordered_cols = (
        ALL_COLUMNS + ["uid"]
        if "uid" not in ALL_COLUMNS
        else ALL_COLUMNS
    )

    df = df[
        [
            c
            for c in ordered_cols
            if c in df.columns
        ]
    ]

    return df


# ---------------------------------------------------------------------------
# Evaluation-data requirement
# ---------------------------------------------------------------------------

def _require_eval_data():

    if not EVAL_AVAILABLE or EVAL_DF is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "No held-out evaluation data found at "
                "data/x_test_sample.csv.gz and "
                "data/y_test_sample.csv. "
                "Dashboard/Performance/Investigation endpoints "
                "need real labeled test data."
            ),
        )


# ---------------------------------------------------------------------------
# Find raw transaction from compressed dataset
#
# IMPORTANT:
# We scan the compressed file in chunks instead of loading the whole
# dataset into memory.
# ---------------------------------------------------------------------------

def _find_raw_transaction(transaction_id: int) -> Optional[pd.Series]:

    if not EVAL_AVAILABLE:
        return None

    for chunk in pd.read_csv(
        _x_test_path,
        compression="gzip",
        chunksize=EVAL_CHUNK_SIZE,
    ):

        match = chunk[
            chunk["TransactionID"] == transaction_id
        ]

        if not match.empty:
            return match.iloc[0]

    return None


def _human_row(
    r: pd.Series,
    result_row: Optional[pd.Series] = None,
) -> Dict[str, Any]:

    if result_row is not None:

        actual_label = int(result_row["isFraud"])
        predicted = int(result_row["predicted"])
        probability = float(result_row["fraud_probability"])
        status = str(result_row["status"])
        correct = bool(result_row["correct"])

    else:

        actual_label = int(r["isFraud"])
        predicted = int(r["predicted"])
        probability = float(r["fraud_probability"])
        status = str(r["status"])
        correct = bool(r["correct"])

    return {
        "transaction_id": (
            int(r["TransactionID"])
            if not pd.isna(r.get("TransactionID"))
            else None
        ),

        "amount": (
            None
            if pd.isna(r.get("TransactionAmt"))
            else round(float(r["TransactionAmt"]), 2)
        ),

        "product_category": (
            None
            if pd.isna(r.get("ProductCD"))
            else str(r["ProductCD"])
        ),

        "card_network": (
            None
            if pd.isna(r.get("card4"))
            else str(r["card4"])
        ),

        "payment_method": (
            None
            if pd.isna(r.get("card6"))
            else str(r["card6"])
        ),

        "device_type": (
            None
            if pd.isna(r.get("DeviceType"))
            else str(r["DeviceType"])
        ),

        "address_region_code": {
            "addr1": (
                None
                if pd.isna(r.get("addr1"))
                else float(r["addr1"])
            ),

            "addr2": (
                None
                if pd.isna(r.get("addr2"))
                else float(r["addr2"])
            ),

            "note": (
                "Anonymized regional code from the dataset, "
                "not a real address."
            ),
        },

        "card_feature_anonymized": {

            "card1": (
                None
                if pd.isna(r.get("card1"))
                else float(r["card1"])
            ),

            "card2": (
                None
                if pd.isna(r.get("card2"))
                else float(r["card2"])
            ),

            "card3": (
                None
                if pd.isna(r.get("card3"))
                else float(r["card3"])
            ),

            "card5": (
                None
                if pd.isna(r.get("card5"))
                else float(r["card5"])
            ),

            "note": (
                "Anonymized card identifiers from the dataset, "
                "not a real card number."
            ),
        },

        "actual_label": (
            "fraud"
            if actual_label == 1
            else "legitimate"
        ),

        "predicted_label": (
            "fraud"
            if predicted == 1
            else "legitimate"
        ),

        "fraud_probability": round(
            probability,
            4,
        ),

        "status": status,

        "correct": correct,
    }


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "ok",
        "threshold": threshold,
        "evaluation_data_available": EVAL_AVAILABLE,
        "evaluation_transactions": (
            int(len(EVAL_DF))
            if EVAL_DF is not None
            else 0
        ),
    }


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

@app.get("/schema")
def schema():

    return {
        "required_fields": REQUIRED_FIELDS,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "numeric_columns": NUMERIC_COLUMNS,
        "total_columns": len(ALL_COLUMNS),
    }


# ---------------------------------------------------------------------------
# Single prediction
# ---------------------------------------------------------------------------

@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(txn: Transaction):

    df = build_dataframe(txn)

    try:

        proba = model.predict_proba(df)[:, 1][0]

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Model inference failed: {e}",
        )

    return PredictionResponse(
        fraud_probability=float(proba),
        is_fraud=bool(proba >= threshold),
        threshold=threshold,
    )


# ---------------------------------------------------------------------------
# Dashboard summary
# ---------------------------------------------------------------------------

@app.get("/dashboard/summary")
def dashboard_summary():

    _require_eval_data()

    y_true = EVAL_DF["isFraud"]
    y_pred = EVAL_DF["predicted"]
    y_proba = EVAL_DF["fraud_probability"]

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
    ).ravel()

    return {

        "dataset_source": (
            "Held-out split of labeled IEEE-CIS training data "
            "(NOT the official unlabeled Kaggle test set)"
        ),

        "model": "XGBoost",

        "threshold": threshold,

        "kpis": {

            "total_transactions": int(
                len(EVAL_DF)
            ),

            "actual_fraud": int(
                y_true.sum()
            ),

            "predicted_fraud": int(
                y_pred.sum()
            ),

            "correct_predictions": int(
                EVAL_DF["correct"].sum()
            ),

            "fraud_rate": round(
                float(y_true.mean()) * 100,
                3,
            ),

            "precision": round(
                precision_score(y_true, y_pred),
                4,
            ),

            "recall": round(
                recall_score(y_true, y_pred),
                4,
            ),

            "f1_score": round(
                f1_score(y_true, y_pred),
                4,
            ),

            "roc_auc": round(
                roc_auc_score(y_true, y_proba),
                4,
            ),

            "pr_auc": round(
                average_precision_score(y_true, y_proba),
                4,
            ),

            "accuracy": round(
                accuracy_score(y_true, y_pred),
                4,
            ),
        },

        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },

        "actual_vs_predicted": {

            "actual_legitimate": int(
                (y_true == 0).sum()
            ),

            "predicted_legitimate": int(
                (y_pred == 0).sum()
            ),

            "actual_fraud": int(
                (y_true == 1).sum()
            ),

            "predicted_fraud": int(
                (y_pred == 1).sum()
            ),
        },

        "class_distribution": {

            "legitimate": int(
                (y_true == 0).sum()
            ),

            "fraud": int(
                (y_true == 1).sum()
            ),
        },
    }


# ---------------------------------------------------------------------------
# Dashboard recent transactions
# ---------------------------------------------------------------------------

@app.get("/dashboard/recent")
def dashboard_recent(
    limit: int = Query(
        20,
        ge=1,
        le=200,
    )
):

    _require_eval_data()

    result_rows = EVAL_DF.head(limit)

    transaction_ids = (
        result_rows["TransactionID"]
        .astype(np.int64)
        .tolist()
    )

    results_by_id = {
        int(row["TransactionID"]): row
        for _, row in result_rows.iterrows()
    }

    output = []

    for chunk in pd.read_csv(
        _x_test_path,
        compression="gzip",
        chunksize=EVAL_CHUNK_SIZE,
    ):

        matches = chunk[
            chunk["TransactionID"].isin(
                transaction_ids
            )
        ]

        for _, raw_row in matches.iterrows():

            tid = int(
                raw_row["TransactionID"]
            )

            output.append(
                _human_row(
                    raw_row,
                    results_by_id[tid],
                )
            )

            if len(output) >= limit:
                return output

    return output


# ---------------------------------------------------------------------------
# Transaction list/search/filter
# ---------------------------------------------------------------------------

@app.get("/transactions")
def list_transactions(
    search: Optional[str] = Query(
        None,
        description="Search by Transaction ID",
    ),

    filter: str = Query(
        "all",
        pattern="^(all|fraud|legitimate|correct|incorrect)$",
    ),

    limit: int = Query(
        50,
        ge=1,
        le=500,
    ),
):

    _require_eval_data()

    df = EVAL_DF

    if search:

        try:

            search_id = int(search)

            df = df[
                df["TransactionID"] == search_id
            ]

        except ValueError:

            df = df.iloc[0:0]

    if filter == "fraud":

        df = df[
            df["isFraud"] == 1
        ]

    elif filter == "legitimate":

        df = df[
            df["isFraud"] == 0
        ]

    elif filter == "correct":

        df = df[
            df["correct"]
        ]

    elif filter == "incorrect":

        df = df[
            ~df["correct"]
        ]

    result_rows = df.head(limit)

    transaction_ids = (
        result_rows["TransactionID"]
        .astype(np.int64)
        .tolist()
    )

    results_by_id = {
        int(row["TransactionID"]): row
        for _, row in result_rows.iterrows()
    }

    output = []

    if not transaction_ids:
        return output

    for chunk in pd.read_csv(
        _x_test_path,
        compression="gzip",
        chunksize=EVAL_CHUNK_SIZE,
    ):

        matches = chunk[
            chunk["TransactionID"].isin(
                transaction_ids
            )
        ]

        for _, raw_row in matches.iterrows():

            tid = int(
                raw_row["TransactionID"]
            )

            output.append(
                _human_row(
                    raw_row,
                    results_by_id[tid],
                )
            )

            if len(output) >= limit:
                return output

    return output


# ---------------------------------------------------------------------------
# Get one transaction
# ---------------------------------------------------------------------------

@app.get("/transactions/{transaction_id}")
def get_transaction(
    transaction_id: int,
):

    _require_eval_data()

    result_match = EVAL_DF[
        EVAL_DF["TransactionID"] == transaction_id
    ]

    if result_match.empty:

        raise HTTPException(
            status_code=404,
            detail=(
                f"Transaction {transaction_id} "
                f"not found in the held-out set."
            ),
        )

    raw_row = _find_raw_transaction(
        transaction_id
    )

    if raw_row is None:

        raise HTTPException(
            status_code=404,
            detail=(
                f"Transaction {transaction_id} "
                f"not found in the dataset."
            ),
        )

    return _human_row(
        raw_row,
        result_match.iloc[0],
    )


# ---------------------------------------------------------------------------
# Analyze one transaction
# ---------------------------------------------------------------------------

@app.post(
    "/transactions/{transaction_id}/analyze"
)
def analyze_transaction(
    transaction_id: int,
):

    _require_eval_data()

    result_match = EVAL_DF[
        EVAL_DF["TransactionID"] == transaction_id
    ]

    if result_match.empty:

        raise HTTPException(
            status_code=404,
            detail=(
                f"Transaction {transaction_id} "
                f"not found in the held-out set."
            ),
        )

    raw_row = _find_raw_transaction(
        transaction_id
    )

    if raw_row is None:

        raise HTTPException(
            status_code=404,
            detail=(
                f"Transaction {transaction_id} "
                f"not found in the dataset."
            ),
        )

    # Convert the raw Series to a one-row DataFrame.
    x = pd.DataFrame(
        [raw_row]
    )

    # Re-run the real trained pipeline.
    proba = float(
        model.predict_proba(x)[:, 1][0]
    )

    is_fraud = proba >= threshold

    actual_fraud = bool(
        result_match.iloc[0]["isFraud"] == 1
    )

    if actual_fraud and is_fraud:

        status = "true_positive"

    elif not actual_fraud and not is_fraud:

        status = "true_negative"

    elif not actual_fraud and is_fraud:

        status = "false_positive"

    else:

        status = "false_negative"

    return {

        "transaction_id": transaction_id,

        "fraud_probability": round(
            proba,
            4,
        ),

        "threshold": threshold,

        "model_prediction": (
            "fraud"
            if is_fraud
            else "legitimate"
        ),

        "actual_label": (
            "fraud"
            if actual_fraud
            else "legitimate"
        ),

        "status": status,

        "correct": status in (
            "true_positive",
            "true_negative",
        ),
    }


# ---------------------------------------------------------------------------
# Model performance
# ---------------------------------------------------------------------------

@app.get("/performance")
def performance():

    _require_eval_data()

    y_true = EVAL_DF["isFraud"]
    y_pred = EVAL_DF["predicted"]
    y_proba = EVAL_DF["fraud_probability"]

    fpr, tpr, _ = roc_curve(
        y_true,
        y_proba,
    )

    prec, rec, _ = precision_recall_curve(
        y_true,
        y_proba,
    )

    # Downsample curve points for a lighter frontend payload.
    def _downsample(
        a,
        b,
        n=100,
    ):

        if len(a) <= n:

            return list(a), list(b)

        idx = np.linspace(
            0,
            len(a) - 1,
            n,
        ).astype(int)

        return (
            list(np.array(a)[idx]),
            list(np.array(b)[idx]),
        )

    fpr_s, tpr_s = _downsample(
        fpr,
        tpr,
    )

    rec_s, prec_s = _downsample(
        rec,
        prec,
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
    ).ravel()

    return {

        "model_info": {

            "model": "XGBoost",

            "dataset": (
                "IEEE-CIS "
                "(held-out split of labeled training data)"
            ),

            "evaluation": "Held-out test split",

            "threshold": threshold,

            "num_test_transactions": int(
                len(EVAL_DF)
            ),

            "fraud_percentage": round(
                float(y_true.mean()) * 100,
                3,
            ),
        },

        "metrics": {

            "precision": round(
                precision_score(
                    y_true,
                    y_pred,
                ),
                4,
            ),

            "recall": round(
                recall_score(
                    y_true,
                    y_pred,
                ),
                4,
            ),

            "f1_score": round(
                f1_score(
                    y_true,
                    y_pred,
                ),
                4,
            ),

            "roc_auc": round(
                roc_auc_score(
                    y_true,
                    y_proba,
                ),
                4,
            ),

            "pr_auc": round(
                average_precision_score(
                    y_true,
                    y_proba,
                ),
                4,
            ),
        },

        "confusion_matrix": {

            "true_negative": int(tn),

            "false_positive": int(fp),

            "false_negative": int(fn),

            "true_positive": int(tp),
        },

        "roc_curve": {

            "fpr": [
                round(float(x), 4)
                for x in fpr_s
            ],

            "tpr": [
                round(float(x), 4)
                for x in tpr_s
            ],
        },

        "pr_curve": {

            "recall": [
                round(float(x), 4)
                for x in rec_s
            ],

            "precision": [
                round(float(x), 4)
                for x in prec_s
            ],
        },
    }