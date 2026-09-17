from pathlib import Path
import json

import lightgbm as lgb
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "credit_risk_model.lgb"
METADATA_PATH = BASE / "credit_risk_model_metadata.json"
HTML_PATH = BASE / "static" / "index.html"

app = FastAPI(title="Credit Risk Assistant")


def require_file(path: Path, label: str):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {label}: {path.name}. "
            f"Place {path.name} in the same folder as app.py."
        )


require_file(MODEL_PATH, "model")
require_file(METADATA_PATH, "model metadata")
require_file(HTML_PATH, "website")

with METADATA_PATH.open("r", encoding="utf-8") as f:
    metadata = json.load(f)

model = lgb.Booster(model_file=str(MODEL_PATH))

FEATURES = metadata.get("features")
THRESHOLD = float(metadata.get("threshold"))

if not FEATURES or not isinstance(FEATURES, list):
    raise ValueError("credit_risk_model_metadata.json does not contain a valid 'features' list.")

GRADE_MAP = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
EMP_MAP = {
    "< 1 year": 0,
    "1 year": 1,
    "2 years": 2,
    "3 years": 3,
    "4 years": 4,
    "5 years": 5,
    "6 years": 6,
    "7 years": 7,
    "8 years": 8,
    "9 years": 9,
    "10+ years": 10,
}


def encode_sub_grade(sg: str) -> int:
    sg = sg.strip().upper()
    if len(sg) < 2 or sg[0] not in GRADE_MAP:
        raise ValueError("Invalid sub-grade.")
    return (ord(sg[0]) - ord("A")) * 5 + int(sg[1:])


class Applicant(BaseModel):
    loan_amnt: float
    int_rate: float
    installment: float
    grade: str
    sub_grade: str
    emp_length: str
    annual_inc: float
    dti: float
    fico_range_low: int
    fico_range_high: int
    revol_util: float
    revol_bal: float
    open_acc: int
    total_acc: int
    pub_rec: int
    delinq_2yrs: int
    inq_last_6mths: int
    mort_acc: int
    pub_rec_bankruptcies: int
    total_rev_hi_lim: float
    avg_cur_bal: float
    bc_util: float
    pct_tl_nvr_dlq: float
    num_actv_bc_tl: int
    num_actv_rev_tl: int


@app.get("/", response_class=HTMLResponse)
def home():
    return HTML_PATH.read_text(encoding="utf-8")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "threshold": THRESHOLD,
        "feature_count": len(FEATURES),
    }


@app.get("/model-info")
def model_info():
    metrics = metadata.get("test_metrics", {})
    return {
        "threshold": THRESHOLD,
        "feature_count": len(FEATURES),
        "model": metadata.get("model", "LightGBM"),
        "auc_roc": metrics.get("auc_roc"),
    }


@app.post("/predict")
def predict(a: Applicant):
    try:
        # Basic input validation. These constraints prevent obviously invalid
        # combinations from being sent to the model.
        if a.fico_range_high < a.fico_range_low:
            raise ValueError("FICO high cannot be lower than FICO low.")
        if a.loan_amnt <= 0:
            raise ValueError("Loan amount must be greater than zero.")
        if a.annual_inc < 0:
            raise ValueError("Annual income cannot be negative.")

        row = {
            "loan_amnt": a.loan_amnt,
            "int_rate": a.int_rate,
            "installment": a.installment,
            "annual_inc": a.annual_inc,
            "dti": a.dti,
            "fico_range_low": a.fico_range_low,
            "fico_range_high": a.fico_range_high,
            "revol_util": a.revol_util,
            "revol_bal": a.revol_bal,
            "open_acc": a.open_acc,
            "total_acc": a.total_acc,
            "pub_rec": a.pub_rec,
            "emp_length": EMP_MAP[a.emp_length],
            "grade": GRADE_MAP[a.grade],
            "sub_grade": encode_sub_grade(a.sub_grade),
            "delinq_2yrs": a.delinq_2yrs,
            "inq_last_6mths": a.inq_last_6mths,
            "mort_acc": a.mort_acc,
            "pub_rec_bankruptcies": a.pub_rec_bankruptcies,
            "total_rev_hi_lim": a.total_rev_hi_lim,
            "avg_cur_bal": a.avg_cur_bal,
            "bc_util": a.bc_util,
            "pct_tl_nvr_dlq": a.pct_tl_nvr_dlq,
            "num_actv_bc_tl": a.num_actv_bc_tl,
            "num_actv_rev_tl": a.num_actv_rev_tl,
        }

        # Force EXACTLY the same feature order used when the corrected model
        # was trained.
        missing = [name for name in FEATURES if name not in row]
        if missing:
            raise ValueError(f"Missing model features: {missing}")

        features = pd.DataFrame(
            [[row[name] for name in FEATURES]],
            columns=FEATURES,
        )

        probability = float(
            model.predict(features, validate_features=True)[0]
        )

        return {
            "probability": probability,
            "threshold": THRESHOLD,
            "risk": "Higher Risk" if probability >= THRESHOLD else "Lower Risk",
        }

    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
