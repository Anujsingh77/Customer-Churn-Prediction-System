"""
dashboard.py  –  Streamlit Customer Churn Dashboard
Run: streamlit run dashboard.py
"""
import pickle, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# ── Page config ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme ────────────────────────────────────────────────────────
PRIMARY  = "#5340C9"
DANGER   = "#E24B4A"
SUCCESS  = "#3B6D11"
WARNING  = "#BA7517"
NEUTRAL  = "#888780"

st.markdown("""
<style>
  .metric-card {
    background: #F8F7FF; border-radius: 12px; padding: 1rem 1.25rem;
    border: 1px solid #E0DEFF; margin-bottom: .5rem;
  }
  .metric-label { font-size: 12px; color: #888780; font-weight: 500; }
  .metric-value { font-size: 28px; font-weight: 600; color: #5340C9; }
  .risk-high   { color: #E24B4A; font-weight: 600; }
  .risk-med    { color: #BA7517; font-weight: 600; }
  .risk-low    { color: #3B6D11; font-weight: 600; }
  section[data-testid="stSidebar"] { background: #F4F2FF; }
</style>
""", unsafe_allow_html=True)

# ── Load data ────────────────────────────────────────────────────
@st.cache_data
def load_data():
    if not os.path.exists("outputs/customers_scored.csv"):
        st.error("Run `python churn_pipeline.py` first to generate data.")
        st.stop()
    return pd.read_csv("outputs/customers_scored.csv")

@st.cache_resource
def load_model():
    if not os.path.exists("models/best_model.pkl"):
        return None
    with open("models/best_model.pkl", "rb") as f:
        return pickle.load(f)

df    = load_data()
model_bundle = load_model()

# ── Sidebar filters ──────────────────────────────────────────────
st.sidebar.header("🔍 Filters")
risk_filter = st.sidebar.multiselect(
    "Risk segment", ["High risk", "Medium risk", "Low risk"],
    default=["High risk", "Medium risk", "Low risk"])
contract_filter = st.sidebar.multiselect(
    "Contract type", df["contract"].unique().tolist(),
    default=df["contract"].unique().tolist())
internet_filter = st.sidebar.multiselect(
    "Internet service", df["internet_service"].unique().tolist(),
    default=df["internet_service"].unique().tolist())
tenure_range = st.sidebar.slider("Tenure range (months)", 1, 72, (1, 72))
charge_range = st.sidebar.slider("Monthly charges ($)", 10, 130, (10, 130))

# Filter
fdf = df[
    df["risk_segment"].isin(risk_filter) &
    df["contract"].isin(contract_filter) &
    df["internet_service"].isin(internet_filter) &
    df["tenure"].between(*tenure_range) &
    df["monthly_charges"].between(*charge_range)
].copy()

# ── Header ───────────────────────────────────────────────────────
st.title("📉 Customer Churn Intelligence Dashboard")
st.caption(f"Showing **{len(fdf):,}** of **{len(df):,}** customers after filters · Model: {model_bundle['name'] if model_bundle else 'N/A'}")
st.markdown("---")

# ── KPI row ──────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("Total customers",  f"{len(fdf):,}")
with c2:
    actual_churn = fdf["churn"].mean() if "churn" in fdf.columns else 0
    st.metric("Actual churn rate", f"{actual_churn:.1%}")
with c3:
    high_risk = (fdf["risk_segment"] == "High risk").sum()
    st.metric("High-risk customers", f"{high_risk:,}")
with c4:
    rev_risk = (fdf["churn_prob"] * fdf["monthly_charges"]).sum()
    st.metric("Monthly revenue at risk", f"${rev_risk:,.0f}")
with c5:
    avg_prob = fdf["churn_prob"].mean()
    st.metric("Avg churn probability", f"{avg_prob:.1%}")

st.markdown("---")

# ── Tabs ─────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🎯 Risk Segments", "🔮 Predict Customer", "📋 Customer List"])

# ─ Tab 1: Overview ───────────────────────────────────────────────
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.hist(fdf[fdf["churn"]==0]["churn_prob"], bins=25, alpha=0.65,
                color=PRIMARY, label="Retained", edgecolor="white")
        ax.hist(fdf[fdf["churn"]==1]["churn_prob"], bins=25, alpha=0.65,
                color=DANGER, label="Churned", edgecolor="white")
        ax.axvline(0.5, color="black", linestyle="--", linewidth=1)
        ax.set_title("Predicted churn probability distribution", fontweight="bold")
        ax.set_xlabel("P(churn)")
        ax.legend(frameon=False)
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ct = fdf.groupby("contract")["churn_prob"].mean().sort_values()
        colors_c = [DANGER if v > 0.35 else WARNING if v > 0.20 else SUCCESS for v in ct.values]
        ax.barh(ct.index, ct.values*100, color=colors_c, height=0.45)
        ax.set_title("Avg churn probability by contract", fontweight="bold")
        ax.set_xlabel("% probability")
        ax.spines[["top","right"]].set_visible(False)
        for i, v in enumerate(ct.values):
            ax.text(v*100+0.3, i, f"{v:.1%}", va="center", fontsize=10)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    col3, col4 = st.columns(2)
    with col3:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        sample = fdf.sample(min(500, len(fdf)), random_state=1)
        sc = ax.scatter(sample["tenure"], sample["monthly_charges"],
                        c=sample["churn_prob"], cmap="RdYlGn_r",
                        s=20, alpha=0.7, vmin=0, vmax=1)
        plt.colorbar(sc, ax=ax, label="P(churn)")
        ax.set_title("Tenure vs charges — churn probability", fontweight="bold")
        ax.set_xlabel("Tenure (months)")
        ax.set_ylabel("Monthly charges ($)")
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col4:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ct2 = fdf.groupby("internet_service")["churn_prob"].mean().sort_values()
        colors_i = [DANGER if v > 0.35 else WARNING if v > 0.20 else SUCCESS for v in ct2.values]
        ax.bar(ct2.index, ct2.values*100, color=colors_i, width=0.45)
        ax.set_title("Avg churn probability by internet", fontweight="bold")
        ax.set_ylabel("% probability")
        ax.spines[["top","right"]].set_visible(False)
        for i, v in enumerate(ct2.values):
            ax.text(i, v*100+0.3, f"{v:.1%}", ha="center", fontsize=10)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

# ─ Tab 2: Risk segments ──────────────────────────────────────────
with tab2:
    col1, col2, col3 = st.columns(3)
    for col, seg, color, icon in [
        (col1, "High risk",   DANGER,  "🔴"),
        (col2, "Medium risk", WARNING, "🟡"),
        (col3, "Low risk",    SUCCESS, "🟢"),
    ]:
        seg_df = fdf[fdf["risk_segment"] == seg]
        with col:
            st.markdown(f"### {icon} {seg}")
            st.metric("Count",             f"{len(seg_df):,}")
            st.metric("Avg churn prob",    f"{seg_df['churn_prob'].mean():.1%}")
            st.metric("Avg tenure",        f"{seg_df['tenure'].mean():.0f} mo")
            st.metric("Avg monthly charge",f"${seg_df['monthly_charges'].mean():.0f}")
            rev = (seg_df["churn_prob"] * seg_df["monthly_charges"]).sum()
            st.metric("Revenue at risk",   f"${rev:,.0f}")

    st.markdown("---")
    # Segment breakdown by contract
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    ct_seg = fdf.groupby(["risk_segment","contract"], observed=True).size().unstack(fill_value=0)
    ct_seg.reindex(["Low risk","Medium risk","High risk"]).plot(
        kind="bar", ax=axes[0], color=[PRIMARY, WARNING, DANGER][:len(ct_seg.columns)],
        edgecolor="white")
    axes[0].set_title("Risk segments by contract type", fontweight="bold")
    axes[0].set_xlabel("")
    axes[0].legend(frameon=False)
    axes[0].spines[["top","right"]].set_visible(False)
    axes[0].tick_params(axis='x', rotation=0)

    ct_int = fdf.groupby(["risk_segment","internet_service"], observed=True).size().unstack(fill_value=0)
    ct_int.reindex(["Low risk","Medium risk","High risk"]).plot(
        kind="bar", ax=axes[1], color=[PRIMARY, WARNING, DANGER][:len(ct_int.columns)],
        edgecolor="white")
    axes[1].set_title("Risk segments by internet service", fontweight="bold")
    axes[1].set_xlabel("")
    axes[1].legend(frameon=False)
    axes[1].spines[["top","right"]].set_visible(False)
    axes[1].tick_params(axis='x', rotation=0)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

# ─ Tab 3: Predict single customer ────────────────────────────────
with tab3:
    st.subheader("🔮 Predict churn for a new customer")
    if model_bundle is None:
        st.warning("Model not found. Run the pipeline first.")
    else:
        model_obj = model_bundle["model"]
        scaler_obj= model_bundle["scaler"]
        features  = model_bundle["features"]
        model_name= model_bundle["name"]

        c1, c2, c3 = st.columns(3)
        with c1:
            tenure = st.slider("Tenure (months)", 1, 72, 12)
            monthly= st.slider("Monthly charges ($)", 18, 120, 65)
            contract= st.selectbox("Contract", ["Month-to-month","One year","Two year"])
        with c2:
            internet= st.selectbox("Internet service", ["Fiber optic","DSL","No"])
            payment = st.selectbox("Payment method", ["Electronic check","Mailed check","Bank transfer","Credit card"])
            senior  = st.checkbox("Senior citizen")
        with c3:
            partner    = st.checkbox("Has partner")
            dependents = st.checkbox("Has dependents")
            num_svc    = st.slider("Number of add-on services", 0, 7, 2)

        if st.button("🔮 Predict churn probability", type="primary"):
            # Build feature row
            total_c = monthly * tenure
            contract_enc_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
            internet_enc_map = {"DSL": 0, "Fiber optic": 1, "No": 2}
            payment_enc_map  = {"Bank transfer": 0, "Credit card": 1, "Electronic check": 2, "Mailed check": 3}

            row = {
                "tenure": tenure, "monthly_charges": monthly,
                "total_charges": total_c, "num_services": num_svc,
                "senior_citizen": int(senior), "partner": int(partner),
                "dependents": int(dependents), "phone_service": 1,
                "multiple_lines": 0, "online_security": 0, "online_backup": 0,
                "device_protection": 0, "tech_support": 0,
                "streaming_tv": 0, "streaming_movies": 0, "paperless_billing": 1,
                "internet_service_enc": internet_enc_map[internet],
                "contract_enc": contract_enc_map[contract],
                "payment_method_enc": payment_enc_map[payment],
                "avg_monthly_spend": total_c / (tenure + 1),
                "charge_per_service": monthly / (num_svc + 1),
                "is_new_customer": int(tenure <= 3),
                "is_long_term": int(tenure >= 48),
                "has_fiber": int(internet == "Fiber optic"),
                "month_to_month": int(contract == "Month-to-month"),
                "electronic_check": int(payment == "Electronic check"),
            }
            X_row = pd.DataFrame([row])[features]
            if model_name == "Logistic Regression":
                X_row = scaler_obj.transform(X_row)
            prob = model_obj.predict_proba(X_row)[0][1]

            st.markdown("---")
            cc1, cc2, cc3 = st.columns(3)
            color = DANGER if prob >= 0.6 else WARNING if prob >= 0.3 else SUCCESS
            risk  = "🔴 High risk" if prob >= 0.6 else "🟡 Medium risk" if prob >= 0.3 else "🟢 Low risk"
            with cc1:
                st.markdown(f"<div style='font-size:42px;font-weight:700;color:{color}'>{prob:.1%}</div>",
                            unsafe_allow_html=True)
                st.caption("Churn probability")
            with cc2:
                st.markdown(f"**Risk level:** {risk}")
                st.markdown(f"**Model:** {model_name}")
            with cc3:
                drivers = []
                if prob >= 0.3:
                    if contract == "Month-to-month":   drivers.append("Month-to-month contract")
                    if internet == "Fiber optic":       drivers.append("Fiber optic service")
                    if payment == "Electronic check":   drivers.append("Electronic check payment")
                    if tenure <= 6:                     drivers.append("New customer (short tenure)")
                    if monthly >= 80:                   drivers.append("High monthly charges")
                if drivers:
                    st.markdown("**Key risk drivers:**")
                    for d in drivers[:4]: st.markdown(f"• {d}")
                else:
                    st.success("No major risk drivers identified.")

# ─ Tab 4: Customer list ──────────────────────────────────────────
with tab4:
    st.subheader("📋 Customer risk table")
    cols_show = ["customer_id","tenure","monthly_charges","contract",
                 "internet_service","num_services","churn_prob","risk_segment","churn"]

    def color_risk(val):
        if val == "High risk":   return "color: #E24B4A; font-weight:600"
        if val == "Medium risk": return "color: #BA7517; font-weight:600"
        return "color: #3B6D11; font-weight:600"

    def color_prob(val):
        if val >= 0.6: return "color: #E24B4A"
        if val >= 0.3: return "color: #BA7517"
        return "color: #3B6D11"

    sort_col = st.selectbox("Sort by", ["churn_prob","monthly_charges","tenure"], index=0)
    top_n    = st.slider("Show top N customers", 20, 200, 50, step=10)

    display_df = fdf[cols_show].sort_values(sort_col, ascending=False).head(top_n)
    styled = (display_df.style
              .applymap(color_risk, subset=["risk_segment"])
              .applymap(color_prob, subset=["churn_prob"])
              .format({"churn_prob": "{:.1%}", "monthly_charges": "${:.0f}"}))
    st.dataframe(styled, use_container_width=True)

    csv = display_df.to_csv(index=False).encode()
    st.download_button("⬇️ Download as CSV", csv, "churn_risk_customers.csv", "text/csv")
