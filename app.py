"""Streamlit app: Will this visitor buy?  (NextGen Institute ML final project)"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from features import RAW_FEATURES  # also makes add_features importable for joblib

BASE = Path(__file__).parent
MODEL_PATH = BASE / "models" / "shopper_pipeline.joblib"
META_PATH = BASE / "models" / "metadata.json"

st.set_page_config(page_title="Will this visitor buy?", page_icon="🛒", layout="centered")


@st.cache_resource
def load_artifacts():
    return joblib.load(MODEL_PATH), json.loads(META_PATH.read_text())


if not (MODEL_PATH.exists() and META_PATH.exists()):
    st.error("Model files not found. Run the notebook first to create `models/shopper_pipeline.joblib` "
             "and `models/metadata.json`, then restart the app.")
    st.stop()

model, meta = load_artifacts()

# ----------------------------------------------------------------- example sessions
EXAMPLES = {
    "Custom (start from average browsing)": dict(adm=2, adm_d=60, inf=0, inf_d=0, prod=25, prod_d=900,
                                                 bounce=2.0, exit=4.0, pv=0.0, special=0.0, weekend=False,
                                                 visitor="Returning_Visitor"),
    "Quick look, about to leave": dict(adm=0, adm_d=0, inf=0, inf_d=0, prod=2, prod_d=40,
                                       bounce=15.0, exit=18.0, pv=0.0, special=0.0, weekend=False,
                                       visitor="New_Visitor"),
    "Deep browsing, engaged": dict(adm=5, adm_d=200, inf=2, inf_d=90, prod=60, prod_d=2500,
                                   bounce=0.5, exit=1.5, pv=30.0, special=0.0, weekend=False,
                                   visitor="Returning_Visitor"),
}

st.title("🛒 Will this visitor buy?")
st.write("Describe a browsing session and the model estimates the chance that the visitor will make a purchase, "
         "then suggests what the website should do.")

example = st.selectbox("Start from an example", list(EXAMPLES))
ex = EXAMPLES[example]
k = example  # changing the example resets the widgets

# ----------------------------------------------------------------- session form
with st.form("session"):
    st.subheader("Pages visited")
    c1, c2, c3 = st.columns(3)
    adm = c1.number_input("Account / admin pages", 0, 100, ex["adm"], key=f"adm{k}")
    inf = c2.number_input("Info / help pages", 0, 100, ex["inf"], key=f"inf{k}")
    prod = c3.number_input("Product pages", 0, 1000, ex["prod"], key=f"prod{k}")

    st.subheader("Time spent (seconds)")
    c1, c2, c3 = st.columns(3)
    adm_d = c1.number_input("On account pages", 0, 100000, ex["adm_d"], key=f"admd{k}")
    inf_d = c2.number_input("On info pages", 0, 100000, ex["inf_d"], key=f"infd{k}")
    prod_d = c3.number_input("On product pages", 0, 200000, ex["prod_d"], key=f"prodd{k}")

    st.subheader("Behaviour signals")
    c1, c2, c3 = st.columns(3)
    bounce = c1.number_input("Bounce rate (%)", 0.0, 100.0, ex["bounce"], step=0.5, key=f"b{k}",
                             help="Share of visits to the pages that left without doing anything else.")
    exit_ = c2.number_input("Exit rate (%)", 0.0, 100.0, ex["exit"], step=0.5, key=f"e{k}",
                            help="Share of page views that were the last in the session.")
    pv = c3.number_input("Page value", 0.0, 1000.0, ex["pv"], step=1.0, key=f"pv{k}",
                         help="Average value of the pages visited before a purchase (from your analytics tool).")

    st.subheader("Visit context")
    c1, c2, c3 = st.columns(3)
    month = c1.selectbox("Month", meta["months"], index=meta["months"].index("Nov") if "Nov" in meta["months"] else 0)
    visitor = c2.selectbox("Visitor type", meta["visitor_types"],
                           index=meta["visitor_types"].index(ex["visitor"]) if ex["visitor"] in meta["visitor_types"] else 0,
                           key=f"v{k}")
    special = c3.selectbox("Closeness to a special day", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                           index=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0].index(ex["special"]),
                           help="0 = not near any special day, 1 = the special day itself.")
    weekend = st.checkbox("Weekend visit", value=ex["weekend"], key=f"w{k}")

    with st.expander("Technical details (optional – defaults are the most common values)"):
        c1, c2 = st.columns(2)
        opts, defs = meta["categorical_options"], meta["categorical_defaults"]
        os_ = c1.selectbox("Operating system code", opts["OperatingSystems"], opts["OperatingSystems"].index(defs["OperatingSystems"]))
        browser = c2.selectbox("Browser code", opts["Browser"], opts["Browser"].index(defs["Browser"]))
        region = c1.selectbox("Region code", opts["Region"], opts["Region"].index(defs["Region"]))
        traffic = c2.selectbox("Traffic source code", opts["TrafficType"], opts["TrafficType"].index(defs["TrafficType"]))

    submitted = st.form_submit_button("Predict", type="primary", use_container_width=True)

# ----------------------------------------------------------------- policy sliders
with st.sidebar:
    st.header("Action policy")
    st.caption("Cut-offs learned from training data. A manager can move them.")
    t_buy = st.slider("No action above", 0.0, 1.0, float(round(meta["default_t_buy"], 2)), 0.01)
    t_chat = st.slider("Chat assistant below", 0.0, 1.0, float(round(meta["default_t_chat"], 2)), 0.01)
    with st.expander("About this model"):
        tm, bm = meta["test_metrics"], meta["baseline_test_metrics"]
        st.write(f"**{meta['model_name']}** trained on {meta['n_train']:,} sessions "
                 f"({meta['purchase_rate']:.1%} end in a purchase).")
        st.write(f"On {meta['n_test']:,} unseen test sessions: ROC-AUC **{tm['roc_auc']:.3f}**, "
                 f"recall **{tm['recall']:.2f}**, precision **{tm['precision']:.2f}** "
                 f"(do-nothing baseline ROC-AUC {bm['roc_auc']:.2f}).")
        st.caption("Data: UCI Online Shoppers Purchasing Intention. A decision-support tool, not a guarantee.")


# ----------------------------------------------------------------- validation + prediction
def validate():
    problems = []
    if adm == 0 and adm_d > 0:
        problems.append("Time on account pages is above 0 but no account pages were visited.")
    if inf == 0 and inf_d > 0:
        problems.append("Time on info pages is above 0 but no info pages were visited.")
    if prod == 0 and prod_d > 0:
        problems.append("Time on product pages is above 0 but no product pages were visited.")
    if bounce > exit_:
        problems.append("Bounce rate cannot be higher than exit rate. Please check the two values.")
    if t_chat >= t_buy:
        problems.append("In the sidebar, the chat cut-off must be lower than the no-action cut-off.")
    return problems


if submitted:
    problems = validate()
    if problems:
        for p in problems:
            st.error(p)
        st.stop()

    row = pd.DataFrame([{
        "Administrative": adm, "Administrative_Duration": adm_d,
        "Informational": inf, "Informational_Duration": inf_d,
        "ProductRelated": prod, "ProductRelated_Duration": prod_d,
        "BounceRates": bounce / 100, "ExitRates": exit_ / 100, "PageValues": pv, "SpecialDay": special,
        "Month": month, "OperatingSystems": os_, "Browser": browser, "Region": region, "TrafficType": traffic,
        "VisitorType": visitor, "Weekend": weekend,
    }])[RAW_FEATURES]

    try:
        p = float(model.predict_proba(row)[0, 1])
    except Exception as exc:  # never crash in front of a user
        st.error(f"Sorry, the prediction failed for this input ({type(exc).__name__}). Please check the values.")
        st.stop()

    st.divider()
    st.metric("Purchase score", f"{p:.0%}")
    st.progress(min(max(p, 0.0), 1.0))

    if p >= t_buy:
        st.success("**No action needed.** This visitor is very likely to buy on their own — a discount would be wasted margin.")
    elif p >= t_chat:
        st.warning("**Show a coupon.** This visitor is undecided: some interest, not yet convinced.")
    else:
        st.info("**Offer the chat assistant.** Low buying intent so far — this visitor may be lost or need help finding something.")

    st.caption("The score is a ranking signal, not an exact probability: the model uses class weights, so scores run "
               "higher than the real purchase rate. Actions are suggestions to be validated with an A/B test.")
else:
    st.caption("Fill in the session details and press **Predict**.")
