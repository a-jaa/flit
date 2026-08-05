import lightgbm as lgb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st

# ============================================================================
# PAGE CONFIG + CUSTOM STYLING
# ============================================================================
st.set_page_config(page_title="CreditGuard | Loan Risk Assistant", page_icon="\U0001F3E6", layout="wide")

st.markdown("""
<style>
    .stApp { background: linear-gradient(180deg, #0b1220 0%, #0f1b2e 100%); }
    #MainMenu, footer, header { visibility: hidden; }

    .hero {
        padding: 2.2rem 2rem; border-radius: 20px; margin-bottom: 1.5rem;
        background: linear-gradient(120deg, #1a2f4f 0%, #16324f 50%, #0d3b3f 100%);
        border: 1px solid rgba(255,255,255,0.08);
    }
    .hero h1 {
        font-size: 2.1rem; font-weight: 800; margin: 0;
        background: linear-gradient(90deg, #7dd3fc, #a7f3d0);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .hero p { color: #94a3b8; margin-top: 0.4rem; font-size: 0.98rem; }

    .card {
        background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px; padding: 1.3rem 1.5rem; margin-bottom: 1rem;
    }
    .reason-card {
        background: rgba(255,255,255,0.05); border-left: 4px solid #38bdf8;
        border-radius: 10px; padding: 0.9rem 1.1rem; margin-bottom: 0.6rem;
    }
    .reason-card.negative { border-left-color: #f87171; }
    .reason-card.positive { border-left-color: #4ade80; }
    .advice-card {
        background: rgba(74, 222, 128, 0.08); border-left: 4px solid #4ade80;
        border-radius: 10px; padding: 0.9rem 1.1rem; margin-bottom: 0.6rem;
    }

    .decision-badge {
        display: inline-block; padding: 0.5rem 1.4rem; border-radius: 999px;
        font-weight: 800; font-size: 1.15rem; letter-spacing: 0.02em;
    }
    .badge-approved { background: rgba(74,222,128,0.15); color: #4ade80; border: 1px solid #4ade80; }
    .badge-review { background: rgba(251,191,36,0.15); color: #fbbf24; border: 1px solid #fbbf24; }
    .badge-rejected { background: rgba(248,113,113,0.15); color: #f87171; border: 1px solid #f87171; }

    .disclaimer {
        font-size: 0.8rem; color: #64748b; border-top: 1px solid rgba(255,255,255,0.08);
        padding-top: 1rem; margin-top: 2rem;
    }
    div[data-testid="stForm"] { background: rgba(255,255,255,0.03); border-radius: 16px;
        padding: 1.5rem; border: 1px solid rgba(255,255,255,0.08); }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>CreditGuard</h1>
    <p>Enter what you know about your finances. We estimate the rest, show you the model's
    reasoning, and tell you exactly what would move the needle.</p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODEL
# ============================================================================
@st.cache_resource
def load_model():
    booster = lgb.Booster(model_file="credit_risk_model.lgb")
    explainer = shap.TreeExplainer(booster)
    return booster, explainer


booster, explainer = load_model()
FEATURE_ORDER = booster.feature_name()

# Threshold tuned during model evaluation (best F1 on the default class).
# Below this predicted-default-probability -> approve; above -> reject/review.
APPROVE_THRESHOLD = 0.35
REVIEW_THRESHOLD = 0.50

# Approximate typical LendingClub interest rate by letter grade -- used only
# to estimate fields the applicant wouldn't know before actually applying.
GRADE_RATE_TABLE = {1: 7.0, 2: 10.5, 3: 13.5, 4: 17.5, 5: 21.0, 6: 26.0, 7: 30.0}


def fico_to_grade(fico):
    if fico >= 750: return 1
    if fico >= 700: return 2
    if fico >= 660: return 3
    if fico >= 620: return 4
    if fico >= 580: return 5
    if fico >= 540: return 6
    return 7


def estimate_installment(principal, annual_rate_pct, months=36):
    r = (annual_rate_pct / 100) / 12
    if r == 0:
        return principal / months
    return principal * r * (1 + r) ** months / ((1 + r) ** months - 1)


# ============================================================================
# REASON TEMPLATES -- only for fields the USER actually provided.
# Derived/estimated fields are never surfaced as "your fault" in the reasons.
# ============================================================================
REASON_TEMPLATES = {
    "loan_amnt": {
        "reject": "The requested loan amount is large relative to your income and profile",
        "approve": "The requested loan amount is reasonable for your profile",
        "advice": "Requesting a smaller amount, or increasing your down payment, reduces this risk factor",
    },
    "annual_inc": {
        "reject": "Income is on the lower side relative to the loan requested",
        "approve": "Income comfortably supports the requested loan",
        "advice": "A co-signer or documented additional income can strengthen this factor",
    },
    "dti": {
        "reject": "Your debt-to-income ratio is high",
        "approve": "Your debt-to-income ratio is healthy",
        "advice": "Paying down existing monthly debts lowers this ratio directly",
    },
    "emp_length": {
        "reject": "A shorter employment history is working against this application",
        "approve": "Your employment history supports this application",
        "advice": "This factor improves naturally with time in a stable job",
    },
    "fico_range_low": {
        "reject": "Your estimated credit score is a limiting factor",
        "approve": "Your estimated credit score is a strong factor in your favor",
        "advice": "On-time payments and lower credit utilization are the fastest ways to raise this",
    },
    "fico_range_high": {
        "reject": "Your estimated credit score is a limiting factor",
        "approve": "Your estimated credit score is a strong factor in your favor",
        "advice": "On-time payments and lower credit utilization are the fastest ways to raise this",
    },
    "revol_util": {
        "reject": "Your reported credit utilization is high",
        "approve": "Your reported credit utilization is in a healthy range",
        "advice": "Utilization under 30% of your available credit typically strengthens applications",
    },
    "delinq_2yrs": {
        "reject": "Recent delinquencies are a significant negative factor",
        "approve": "A clean recent payment history supports this application",
        "advice": "Consistent on-time payments going forward will steadily reduce this factor's weight over time",
    },
    "mort_acc": {
        "reject": "Limited mortgage/installment account history is a minor negative factor",
        "approve": "Your account mix supports this application",
        "advice": "A diverse, well-managed credit mix (not just revolving credit) is viewed favorably",
    },
    "inq_last_6mths": {
        "reject": "Several recent credit inquiries are a negative factor",
        "approve": "Few recent credit inquiries support this application",
        "advice": "Avoid applying for new credit in the months before an important application",
    },
    "pub_rec_bankruptcies": {
        "reject": "A reported bankruptcy is a significant negative factor",
        "approve": "No bankruptcy history supports this application",
        "advice": "This factor's impact fades over time as more recent positive history accumulates",
    },
    "open_acc": {
        "reject": "Your number of open accounts is a minor negative factor",
        "approve": "Your number of open accounts is a minor positive factor",
        "advice": "Avoid opening several new accounts in a short window before applying",
    },
}

USER_PROVIDED_FEATURES = set(REASON_TEMPLATES.keys())


# ============================================================================
# INPUT FORM -- only fields a real applicant would actually know
# ============================================================================
with st.form("application_form"):
    st.subheader("Tell us about your application")
    c1, c2, c3 = st.columns(3)

    with c1:
        loan_amnt = st.number_input("Loan amount requested ($)", 1000, 40000, 15000, step=500)
        annual_inc = st.number_input("Annual income ($)", 10000, 500000, 55000, step=1000)
        monthly_debt = st.number_input("Total monthly debt payments ($)", 0, 20000, 500, step=50,
                                        help="Rent/mortgage, car payments, minimum credit card payments, student loans, etc.")

    with c2:
        emp_length_label = st.selectbox(
            "Employment length",
            ["< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years",
             "6 years", "7 years", "8 years", "9 years", "10+ years"],
            index=2,
        )
        fico = st.slider("Estimated credit score (FICO)", 300, 850, 680,
                          help="Don't know it exactly? A rough estimate is fine.")
        revol_util = st.slider("Credit utilization (%)", 0, 150, 35,
                                help="Roughly what % of your total available credit are you currently using?")

    with c3:
        open_acc = st.number_input("Number of open credit accounts", 0, 40, 8)
        delinq_2yrs = st.number_input("Late payments (30+ days) in the last 2 years", 0, 20, 0)
        inq_last_6mths = st.number_input("Credit inquiries in the last 6 months", 0, 20, 1)
        has_bankruptcy = st.checkbox("I have a bankruptcy on record")
        has_mortgage = st.checkbox("I have a mortgage or auto loan account")

    submitted = st.form_submit_button("Check my application", use_container_width=True)

with st.expander("How we estimate the rest (transparency)"):
    st.write(
        "This model was trained on 25 credit-bureau-level fields. Asking for all 25 up front "
        "isn't realistic -- most applicants don't know their own 'bankcard utilization' or "
        "'percent of trades never delinquent' offhand. We only ask for the fields above, and "
        "estimate the rest using standard relationships: your credit score maps to an estimated "
        "loan grade and interest rate, your income implies a typical available credit line, and "
        "your utilization is applied proportionally across accounts. These estimated fields are "
        "never used in the 'what drove this decision' explanation below -- only the fields you "
        "actually entered are."
    )

# ============================================================================
# PREDICTION PIPELINE
# ============================================================================
if submitted:
    emp_map = {"< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3, "4 years": 4,
               "5 years": 5, "6 years": 6, "7 years": 7, "8 years": 8, "9 years": 9, "10+ years": 10}
    emp_length = emp_map[emp_length_label]

    dti = (monthly_debt * 12 / annual_inc) * 100 if annual_inc > 0 else 0
    dti = min(dti, 100)

    grade = fico_to_grade(fico)
    sub_grade = (grade - 1) * 5 + 3  # middle sub-grade within the estimated letter grade
    int_rate = GRADE_RATE_TABLE[grade]
    installment = estimate_installment(loan_amnt, int_rate)

    fico_low = max(300, fico - 2)
    fico_high = min(850, fico + 4)

    total_rev_hi_lim = max(1000, annual_inc * 0.2)
    revol_bal = (revol_util / 100) * total_rev_hi_lim
    avg_cur_bal = revol_bal / max(open_acc, 1)
    bc_util = revol_util
    total_acc = round(open_acc * 1.6)
    pub_rec = 1 if has_bankruptcy else 0
    pub_rec_bankruptcies = 1 if has_bankruptcy else 0
    mort_acc = 1 if has_mortgage else 0
    pct_tl_nvr_dlq = max(40, 100 - delinq_2yrs * 12)
    num_actv_bc_tl = max(1, round(open_acc * 0.35))
    num_actv_rev_tl = max(1, round(open_acc * 0.45))

    row = {
        "loan_amnt": loan_amnt, "int_rate": int_rate, "installment": installment,
        "annual_inc": annual_inc, "dti": dti, "fico_range_low": fico_low,
        "fico_range_high": fico_high, "revol_util": revol_util, "revol_bal": revol_bal,
        "open_acc": open_acc, "total_acc": total_acc, "pub_rec": pub_rec,
        "emp_length": emp_length, "grade": grade, "sub_grade": sub_grade,
        "delinq_2yrs": delinq_2yrs, "inq_last_6mths": inq_last_6mths, "mort_acc": mort_acc,
        "pub_rec_bankruptcies": pub_rec_bankruptcies, "total_rev_hi_lim": total_rev_hi_lim,
        "avg_cur_bal": avg_cur_bal, "bc_util": bc_util, "pct_tl_nvr_dlq": pct_tl_nvr_dlq,
        "num_actv_bc_tl": num_actv_bc_tl, "num_actv_rev_tl": num_actv_rev_tl,
    }

    X = pd.DataFrame([row])[FEATURE_ORDER]

    raw_default_prob = float(booster.predict(X)[0])

    # Several of the 25 fields the model needs are estimated independently
    # from what the user actually provided (see the transparency panel above).
    # Real applicants' fico/grade/utilization/income are correlated in ways
    # our independent guesses don't preserve, which can push the raw
    # prediction to an unrealistic extreme. Shrink toward the dataset's
    # overall default rate to damp out that artifact rather than presenting
    # a falsely precise number.
    BASE_DEFAULT_RATE = 0.20  # approximate dataset-wide default rate
    SHRINKAGE = 0.6  # weight on the raw model output vs. the base rate
    default_prob = SHRINKAGE * raw_default_prob + (1 - SHRINKAGE) * BASE_DEFAULT_RATE
    approval_prob = 1 - default_prob

    if default_prob < APPROVE_THRESHOLD:
        decision, badge_class = "Likely Approved", "badge-approved"
    elif default_prob < REVIEW_THRESHOLD:
        decision, badge_class = "Manual Review Likely", "badge-review"
    else:
        decision, badge_class = "Likely Rejected", "badge-rejected"

    shap_vals = explainer.shap_values(X)
    shap_vals = shap_vals[1] if isinstance(shap_vals, list) else shap_vals
    contributions = sorted(zip(FEATURE_ORDER, shap_vals[0]), key=lambda x: abs(x[1]), reverse=True)

    reasons = []
    for feat, val in contributions:
        if feat not in USER_PROVIDED_FEATURES:
            continue
        template = REASON_TEMPLATES[feat]
        direction = "reject" if val > 0 else "approve"  # positive SHAP -> pushes toward default
        reasons.append({"feature": feat, "value": float(val), "reason": template[direction], "advice": template["advice"]})
        if len(reasons) == 4:
            break

    # ---------------------------------------------------------------------
    # RESULTS
    # ---------------------------------------------------------------------
    st.markdown("---")
    left, right = st.columns([1, 1.4])

    with left:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=approval_prob * 100,
            number={"suffix": "%", "font": {"size": 42, "color": "#e2e8f0"}},
            title={"text": "Estimated Approval Likelihood", "font": {"size": 15, "color": "#94a3b8"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#475569"},
                "bar": {"color": "#38bdf8"},
                "bgcolor": "rgba(255,255,255,0.03)",
                "steps": [
                    {"range": [0, 50], "color": "rgba(248,113,113,0.25)"},
                    {"range": [50, 65], "color": "rgba(251,191,36,0.25)"},
                    {"range": [65, 100], "color": "rgba(74,222,128,0.25)"},
                ],
            },
        ))
        fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=10),
                           paper_bgcolor="rgba(0,0,0,0)", font={"color": "#e2e8f0"})
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f'<span class="decision-badge {badge_class}">{decision}</span>', unsafe_allow_html=True)

        if decision == "Likely Approved":
            st.balloons()

    with right:
        st.markdown("#### What drove this result")
        for r in reasons:
            cls = "negative" if r["value"] > 0 else "positive"
            st.markdown(f'<div class="reason-card {cls}">{r["reason"]}</div>', unsafe_allow_html=True)

        st.markdown("#### How to improve")
        for r in reasons:
            st.markdown(f'<div class="advice-card">{r["advice"]}</div>', unsafe_allow_html=True)

    with st.expander("Full model detail"):
        st.write(f"Raw model output: **{raw_default_prob:.1%}** \u2192 calibrated estimate shown above: **{default_prob:.1%}**")
        st.caption("The calibrated estimate is shrunk toward the dataset's overall default rate to reduce the impact of fields that were estimated rather than provided.")
        st.write(f"Estimated grade: **{['A','B','C','D','E','F','G'][grade-1]}**, estimated rate: **{int_rate:.1f}%**")
        chart_df = pd.DataFrame(contributions[:10], columns=["feature", "shap_value"])
        st.bar_chart(chart_df.set_index("feature"))

st.markdown(
    '<div class="disclaimer">Educational demo trained on historical Lending Club data '
    '(accepted loans only, predicting default risk). Several inputs above are estimated, '
    'not verified. This is not a real credit decision.</div>',
    unsafe_allow_html=True,
)
