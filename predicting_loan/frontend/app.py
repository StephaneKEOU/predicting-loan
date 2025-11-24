import json
from datetime import datetime

import pandas as pd
import streamlit as st

from core.schema_loader import load_schema
from core.config import FEATURE_SCHEMA, DECISION_THRESHOLD, LOW_MEDIUM_THRESHOLD, MEDIUM_HIGH_THRESHOLD
from core.model_client import ModelClient
from core.db import init_db, log

st.set_page_config(
    page_title="Loan Default — MVP",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load schema and model
schema = load_schema(FEATURE_SCHEMA)
model = ModelClient()
init_db()

# Sidebar
with st.sidebar:
    st.title("💰 Loan Default MVP")
    st.markdown("---")
    
    mode = st.radio(
        "**Mode**",
        ["Single Input", "Batch CSV"],
        help="Choose between single application or batch processing"
    )
    
    st.markdown("---")
    st.markdown("**⚙️ Risk Configuration**")
    
    # Configuration container using info box
    with st.expander("Decision Threshold", expanded=False):
        st.metric("Reject Above", f"{DECISION_THRESHOLD:.2%}")
        st.caption("Applications with PD above this threshold are recommended for rejection")
    
    with st.expander("Risk Band Thresholds", expanded=False):
        st.caption(f"**Low Risk:** < {LOW_MEDIUM_THRESHOLD:.2%}")
        st.caption(f"**Medium Risk:** {LOW_MEDIUM_THRESHOLD:.2%} - {MEDIUM_HIGH_THRESHOLD:.2%}")
        st.caption(f"**High Risk:** ≥ {MEDIUM_HIGH_THRESHOLD:.2%}")

# Title
st.title("🏦 Loan Default Risk Prediction")
st.caption("Predict the probability of loan default using machine learning")

def coerce_and_order(df: pd.DataFrame) -> pd.DataFrame:
    cols = []
    for f in schema["features"]:
        name = f["name"]
        dtype = f.get("dtype", "float")
        default = f.get("default", None)

        if name not in df.columns:
            df[name] = default

        if dtype in ("float", "int"):
            df[name] = pd.to_numeric(df[name], errors="coerce")
        elif dtype == "category":
            df[name] = df[name].astype("string")
        else:
            df[name] = df[name].astype("string")

        cols.append(name)
    return df[cols]

if mode == "Single Input":
    st.subheader("📝 Single Applicant")
    
    # Input form in columns
    c1, c2 = st.columns(2)
    values = {}
    for i, f in enumerate(schema["features"]):
        target = c1 if i % 2 == 0 else c2
        name = f["name"]
        dtype = f.get("dtype", "float")
        default = f.get("default", 0)

        if dtype == "int":
            values[name] = target.number_input(name, value=int(default or 0), step=1)
        elif dtype == "float":
            values[name] = target.number_input(name, value=float(default or 0.0))
        elif dtype == "category":
            choices = f.get("choices", [])
            idx = 0
            if choices and default in choices:
                idx = choices.index(default)
            values[name] = target.selectbox(name, choices, index=idx if choices else 0)
        else:
            values[name] = target.text_input(name, value=str(default or ""))

    # Predict button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_clicked = st.button("Predict Risk", use_container_width=True, type="primary")
    
    if predict_clicked:
        with st.spinner("🔍 Analyzing risk profile..."):
            X = coerce_and_order(pd.DataFrame([values]))
            probs, labels, risk_bands = model.predict(X)
            prob = float(probs[0])
            label = int(labels[0])
            risk_band = str(risk_bands[0])
        
        st.markdown("---")
        
        # Determine if result is good or bad
        is_good_result = label == 0
        
        # Results display
        st.subheader("📊 Loan Decision Results")
        
        # Main decision section
        col_main, col_action = st.columns([2, 1])
        
        with col_main:
            # PD display with explanation
            if is_good_result:
                st.success(f"""
                **Probability of Default (PD)**
                
                # {prob:.2%}
                
                *The chance this borrower will fail to repay the loan*
                
                **Risk Level:** ✅ Low Risk
                """)
            else:
                st.error(f"""
                **Probability of Default (PD)**
                
                # {prob:.2%}
                
                *The chance this borrower will fail to repay the loan*
                
                **Risk Level:** ❌ High Risk
                """)
        
        with col_action:
            st.markdown("**🎯 Recommended Action**")
            
            # Actionable recommendation based on risk
            if is_good_result:
                st.success("**✅ APPROVE**\n\nThis applicant has a low risk profile and is likely to repay the loan.")
            else:
                st.error("**❌ REJECT**\n\nThis applicant has a high risk of default. Consider requiring additional collateral or rejecting the application.")
            
            # Risk band with explanation
            st.markdown("**Risk Category**")
            risk_band_lower = risk_band.lower()
            if risk_band_lower == "low":
                st.success(f"✅ **{risk_band} Risk**")
                st.caption(f"PD < {LOW_MEDIUM_THRESHOLD:.0%} - Safe to approve")
            elif risk_band_lower == "medium":
                st.warning(f"⚠️ **{risk_band} Risk**")
                st.caption(f"PD {LOW_MEDIUM_THRESHOLD:.0%}-{MEDIUM_HIGH_THRESHOLD:.0%} - Review carefully")
            else:
                st.error(f"❌ **{risk_band} Risk**")
                st.caption(f"PD ≥ {MEDIUM_HIGH_THRESHOLD:.0%} - High default risk")
        
        # Understanding the Results in expander
        with st.expander("📖 Understanding the Results", expanded=False):
            col_explain1, col_explain2, col_explain3 = st.columns(3)
            
            with col_explain1:
                st.markdown("**📊 Probability of Default (PD)**")
                st.caption(f"**Value:** {prob:.2%}")
                st.caption("**What it means:** The percentage chance that this borrower will not repay the loan.")
                st.caption("**Who uses it:** Risk managers and underwriters use PD to assess credit risk and set interest rates.")
            
            with col_explain2:
                st.markdown("**🏷️ Risk Band**")
                st.caption(f"**Value:** {risk_band} Risk")
                st.caption("**What it means:** A quick categorization (Low/Medium/High) based on the PD value.")
                st.caption("**Who uses it:** Loan officers use this for quick decision-making and portfolio management.")
            
            with col_explain3:
                st.markdown("**✅ Binary Decision**")
                risk_text = "Approve" if is_good_result else "Reject"
                st.caption(f"**Recommendation:** {risk_text}")
                st.caption("**What it means:** A clear approve/reject recommendation based on the risk threshold.")
                st.caption("**Who uses it:** Bank managers use this for final loan approval decisions.")
        
        # Thresholds info
        with st.expander("⚙️ How Risk Bands Are Determined", expanded=False):
            st.markdown("**Risk Classification Thresholds:**")
            st.write(f"- **Low Risk:** PD < {LOW_MEDIUM_THRESHOLD:.2%} - Safe borrowers, approve with standard terms")
            st.write(f"- **Medium Risk:** PD {LOW_MEDIUM_THRESHOLD:.2%} - {MEDIUM_HIGH_THRESHOLD:.2%} - Moderate risk, review carefully, may require higher interest rate")
            st.write(f"- **High Risk:** PD ≥ {MEDIUM_HIGH_THRESHOLD:.2%} - High default risk, consider rejecting or require collateral")
            st.write(f"- **Decision Threshold:** {DECISION_THRESHOLD:.2%} - Applications above this threshold are recommended for rejection")
        
        # Technical details expander
        with st.expander("🔧 Technical Details", expanded=False):
            col_tech1, col_tech2 = st.columns(2)
            
            with col_tech1:
                st.markdown("**Model Output:**")
                st.write(f"- **PD:** {prob:.4f} ({prob:.2%})")
                st.write(f"- **Binary Label:** {label} ({'High Risk' if label == 1 else 'Low Risk'})")
                st.write(f"- **Risk Band:** {risk_band}")
            
            with col_tech2:
                st.markdown("**Thresholds:**")
                st.write(f"- **Decision Threshold:** {DECISION_THRESHOLD:.2%}")
                st.write(f"- **Low < {LOW_MEDIUM_THRESHOLD:.0%}**")
                st.write(f"- **Medium:** {LOW_MEDIUM_THRESHOLD:.0%} - {MEDIUM_HIGH_THRESHOLD:.0%}")
                st.write(f"- **High ≥ {MEDIUM_HIGH_THRESHOLD:.0%}**")
            
            st.markdown("**Input Features:**")
            st.dataframe(X.assign(prob_default=probs, pred_label=labels, risk_band=risk_bands), use_container_width=True)
        
        log(datetime.utcnow().isoformat(), "single", json.dumps(values), prob, label)

else:
    st.subheader("📦 Batch CSV Processing")
    st.markdown("Upload a CSV file to process multiple loan applications at once.")
    
    up = st.file_uploader(
        "Upload CSV",
        type=["csv"],
        help="Upload a CSV file containing applicant data"
    )
    
    if up is not None:
        raw = pd.read_csv(up)
        
        # Preview section
        st.markdown("### 📋 Preview")
        st.caption(f"Total records: {len(raw)}")
        st.dataframe(raw.head(10), use_container_width=True)
        
        X = coerce_and_order(raw.copy())
        
        st.markdown("---")
        
        if st.button("Process Batch", use_container_width=True, type="primary"):
            with st.spinner("⚙️ Processing applications..."):
                probs, labels, risk_bands = model.predict(X)
                out = X.assign(prob_default=probs, pred_label=labels, risk_band=risk_bands)
            
            st.success(f"✅ Batch scoring complete! Processed {len(out)} applications.")
            
            # Summary metrics in columns
            st.markdown("### 📊 Portfolio Summary")
            
            high_risk = sum(labels)
            low_risk = len(out) - high_risk
            avg_prob = float(probs.mean())
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Applications", len(out))
            
            with col2:
                st.metric("✅ Approve", low_risk, delta=f"{low_risk/len(out):.1%} of portfolio", delta_color="normal")
                st.caption("Low risk applications")
            
            with col3:
                st.metric("❌ Reject", high_risk, delta=f"{high_risk/len(out):.1%} of portfolio", delta_color="inverse")
                st.caption("High risk applications")
            
            with col4:
                st.metric("Average PD", f"{avg_prob:.2%}")
                st.caption("Portfolio risk level")
            
            # Action summary - more compact
            st.markdown("---")
            st.markdown("**🎯 Quick Action Summary**")
            
            col_action1, col_action2 = st.columns(2)
            
            with col_action1:
                st.info(f"**✅ Recommended for Approval: {low_risk} applications**\n\nThese applications have a low probability of default and can be approved with standard terms.")
            
            with col_action2:
                st.warning(f"**⚠️ Requires Review: {high_risk} applications**\n\nThese applications have a high risk of default. Consider rejecting or requiring additional collateral.")
            
            st.markdown("---")
            
            # Results table with explanation
            st.markdown("### 📋 Detailed Results")
            st.caption("💡 **Column Guide:** `prob_default` = Probability of Default, `pred_label` = 0 (Approve) or 1 (Reject), `risk_band` = Risk category")
            
            # Add color coding helper
            st.info("💡 **Tip:** Sort by `prob_default` to see highest risk applications first, or filter by `risk_band` to review specific risk categories.")
            
            st.dataframe(out.head(20), use_container_width=True)
            
            # Download button
            st.markdown("---")
            st.download_button(
                "Download Scored CSV",
                out.to_csv(index=False).encode("utf-8"),
                "scored.csv",
                "text/csv",
                use_container_width=True,
                type="primary"
            )
            
            # Log all predictions
            for _, r in out.iterrows():
                payload = json.dumps({k: r[k] for k in X.columns})
                log(datetime.utcnow().isoformat(), "batch", payload, float(r["prob_default"]), int(r["pred_label"]))
