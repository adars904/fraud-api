# Fraud Detection API

Serves your trained fraud-detection pipeline (UID feature engineering →
imputation → encoding → feature selection → XGBoost) via FastAPI.

## What's in the pipeline

`model/model.pkl` is a 5-stage `sklearn.pipeline.Pipeline`:

1. **`uid_features`** – custom `UIDFeatureTransformer` (from
   `features/feature_construction.py`). Builds `uid_TransactionAmt_mean`,
   `uid_TransactionAmt_std`, `Amt_to_mean_ratio` from a synthetic `uid`
   (`card1_card2_card3_card5_addr1_addr2`).
2. **`missing_values`** – median/most-frequent imputation.
3. **`encoding`** – one-hot + ordinal encoding.
4. **`feature_selection`** – `SelectFromModel` (470 → 235 features).
5. **`classifier`** – `XGBClassifier`.

`model/threshold.pkl` holds the decision cutoff (currently `0.81`) applied
to the predicted fraud probability to produce a binary `is_fraud` flag.

⚠️ **Important:** `features/feature_construction.py` must stay at the
project root, importable as `features.feature_construction` — that exact
path is baked into the pickle file itself. Moving or renaming it will break
unpickling.

⚠️ **Version pin:** the model was trained with **scikit-learn 1.7.2**.
Newer sklearn versions (e.g. 1.8.x) changed internal `SimpleImputer`
attributes and will raise `AttributeError` at inference time. Keep the
pinned version in `requirements.txt`.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Interactive docs: http://localhost:8000/docs

## Endpoints

### `GET /health`
Basic liveness check; also returns the configured threshold.

### `GET /schema`
Returns the full list of raw columns the model accepts (required fields,
plus categorical/numeric column lists) — use this to see what you can pass
in `extra_fields`.

### `POST /predict`
Scores one transaction.

**Required fields:** `TransactionAmt`, `card1` (needed to build the `uid`
and derived features). `card2`, `card3`, `card5`, `addr1`, `addr2` are
optional but strongly recommended — missing values just become `"nan"` in
the `uid` string, which groups the transaction with other missing-value
rows (see the note in `feature_construction.py`).

Any other original column (`V1`-`V339`, `C1`-`C14`, `D1`-`D15`, `M1`-`M9`,
`id_01`-`id_38`, `ProductCD`, `card4`, `card6`, `dist1`, `dist2`,
`DeviceType`, `DeviceInfo`, `TransactionID`, `TransactionDT`, ...) can be
passed inside `extra_fields`. Anything you omit is treated as missing and
handled by the pipeline's built-in imputers.

### Dashboard / Investigation / Performance (real held-out evaluation)

`data/x_test_sample.csv` + `data/y_test_sample.csv` are a genuine held-out
split of the labeled IEEE-CIS training data (carved out by your original
`main.py`, not the official unlabeled Kaggle test set — that one has no
`isFraud` labels at all). These endpoints score that data once at startup
and serve real numbers, never invented ones:

- `GET /dashboard/summary` — KPIs, confusion matrix, actual-vs-predicted, class distribution
- `GET /dashboard/recent?limit=20` — recent transactions table
- `GET /transactions?search=&filter=all|fraud|legitimate|correct|incorrect&limit=50` — search/filter the held-out set
- `GET /transactions/{transaction_id}` — human-readable detail for one transaction (anonymized ML columns are labeled honestly, e.g. `card1` is never called "Card Number")
- `POST /transactions/{transaction_id}/analyze` — re-runs that row through the live pipeline and compares prediction vs. actual label
- `GET /performance` — precision/recall/F1/ROC-AUC/PR-AUC, ROC curve points, PR curve points, confusion matrix, model info

If `data/x_test_sample.csv` / `data/y_test_sample.csv` are missing, these
endpoints return `503` with a clear message rather than fabricating numbers.

**Example request:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "TransactionAmt": 125.75,
    "card1": 13553,
    "card2": 150.0,
    "card3": 185.0,
    "card5": 226.0,
    "addr1": 315.0,
    "addr2": 87.0,
    "extra_fields": {
      "ProductCD": "W",
      "card4": "visa",
      "card6": "debit",
      "DeviceType": "mobile",
      "DeviceInfo": "iOS Device",
      "V1": 1.0,
      "C1": 3.0
    }
  }'
```

**Example response:**

```json
{
  "fraud_probability": 0.0122,
  "is_fraud": false,
  "threshold": 0.81
}
```

## Notes / things worth double-checking

- I reverse-engineered the raw input schema (426 columns) directly from the
  fitted pipeline's `feature_names_in_` attributes — it matches the
  IEEE-CIS Fraud Detection dataset schema (`TransactionAmt`, `card1-6`,
  `addr1/2`, `C1-14`, `D1-15`, `M1-9`, `V1-339`, `id_01-38`, `DeviceType`,
  `DeviceInfo`). Double check this against whatever system produces
  transactions for you in production — field names/types must match
  exactly.
- The `uid` construction concatenates `card2`/`card3`/`card5`/`addr1`/`addr2`
  as strings, so missing values become the literal text `"nan"`. This is
  expected behavior carried over from the training code, not a bug I
  introduced — see the comment in `feature_construction.py`.
- No auth/rate-limiting is included — add these before exposing publicly.
