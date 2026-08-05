import streamlit as st
import lightgbm as lgb
import pandas as pd

st.set_page_config(page_title="Credit Risk Assistant", layout="centered")

@st.cache_resource
def load_model():
    return lgb.Booster(model_file="credit_risk_model.lgb")

model = load_model()

GRADE_MAP = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}
EMP_MAP = {
    "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3,
    "4 years": 4, "5 years": 5, "6 years": 6,
    "7 years": 7, "8 years": 8, "9 years": 9, "10+ years": 10
}

def encode_sub_grade(sg):
    letter = sg[0]
    number = int(sg[1])
    return (ord(letter) - ord("A")) * 5 + number

st.title("Credit Risk Assistant")
st.caption("Enter your loan and financial details to estimate default risk.")

st.subheader("Loan Details")
col1, col2 = st.columns(2)
with col1:
    loan_amnt   = st.number_input("Loan amount ($)", min_value=500, max_value=40000, value=10000, step=500)
    int_rate    = st.number_input("Interest rate (%)", min_value=1.0, max_value=35.0, value=12.0, step=0.1)
    installment = st.number_input("Monthly installment ($)", min_value=10.0, max_value=2000.0, value=250.0, step=5.0)
    grade       = st.selectbox("Loan grade", list(GRADE_MAP.keys()))
with col2:
    sub_grade  = st.selectbox("Sub-grade", [f"{g}{n}" for g in "ABCDEFG" for n in range(1, 6)])
    emp_length = st.selectbox("Employment length", list(EMP_MAP.keys()))
    annual_inc = st.number_input("Annual income ($)", min_value=0, max_value=500000, value=60000, step=1000)
    dti        = st.number_input("Debt-to-income ratio (%)", min_value=0.0, max_value=100.0, value=15.0, step=0.1)

st.subheader("Credit Profile")
col3, col4 = st.columns(2)
with col3:
    fico_low   = st.number_input("FICO score (low)", min_value=300, max_value=850, value=680)
    fico_high  = st.number_input("FICO score (high)", min_value=300, max_value=850, value=684)
    revol_util = st.number_input("Revolving utilization (%)", min_value=0.0, max_value=150.0, value=40.0, step=0.1)
    revol_bal  = st.number_input("Revolving balance ($)", min_value=0, max_value=500000, value=10000, step=500)
    open_acc   = st.number_input("Open credit accounts", min_value=0, max_value=80, value=8)
with col4:
    total_acc      = st.number_input("Total credit accounts", min_value=0, max_value=150, value=20)
    pub_rec        = st.number_input("Public records", min_value=0, max_value=20, value=0)
    delinq_2yrs    = st.number_input("Delinquencies (last 2 yrs)", min_value=0, max_value=30, value=0)
    inq_last_6mths = st.number_input("Credit inquiries (last 6 mo)", min_value=0, max_value=30, value=1)
    mort_acc       = st.number_input("Mortgage accounts", min_value=0, max_value=30, value=0)

st.subheader("Additional Info")
col5, col6 = st.columns(2)
with col5:
    pub_rec_bankruptcies = st.number_input("Public record bankruptcies", min_value=0, max_value=10, value=0)
    total_rev_hi_lim     = st.number_input("Total revolving credit limit ($)", min_value=0, max_value=500000, value=30000, step=500)
    avg_cur_bal          = st.number_input("Avg current balance ($)", min_value=0, max_value=500000, value=5000, step=500)
with col6:
    bc_util         = st.number_input("Bankcard utilization (%)", min_value=0.0, max_value=200.0, value=50.0, step=0.1)
    pct_tl_nvr_dlq  = st.number_input("% accounts never delinquent", min_value=0.0, max_value=100.0, value=95.0, step=0.1)
    num_actv_bc_tl  = st.number_input("Active bankcard accounts", min_value=0, max_value=30, value=3)
    num_actv_rev_tl = st.number_input("Active revolving accounts", min_value=0, max_value=30, value=5)

if st.button("Predict Risk", use_container_width=True):
    features = pd.DataFrame([{
        "loan_amnt":            loan_amnt,
        "int_rate":             int_rate,
        "installment":          installment,
        "annual_inc":           annual_inc,
        "dti":                  dti,
        "fico_range_low":       fico_low,
        "fico_range_high":      fico_high,
        "revol_util":           revol_util,
        "revol_bal":            revol_bal,
        "open_acc":             open_acc,
        "total_acc":            total_acc,
        "pub_rec":              pub_rec,
        "emp_length":           EMP_MAP[emp_length],
        "grade":                GRADE_MAP[grade],
        "sub_grade":            encode_sub_grade(sub_grade),
        "delinq_2yrs":          delinq_2yrs,
        "inq_last_6mths":       inq_last_6mths,
        "mort_acc":             mort_acc,
        "pub_rec_bankruptcies": pub_rec_bankruptcies,
        "total_rev_hi_lim":     total_rev_hi_lim,
        "avg_cur_bal":          avg_cur_bal,
        "bc_util":              bc_util,
        "pct_tl_nvr_dlq":       pct_tl_nvr_dlq,
        "num_actv_bc_tl":       num_actv_bc_tl,
        "num_actv_rev_tl":      num_actv_rev_tl,
    }])

    prob = model.predict(features)[0]
    threshold = 0.45

    st.divider()
    if prob >= threshold:
        st.error("**High Risk — Likely Default**")
    else:
        st.success("**Low Risk — Likely to Repay**")

    st.metric("Default Probability", f"{prob:.1%}")
    st.progress(float(prob))
    st.caption("Threshold: 0.45 — scores above this are flagged as high risk.")