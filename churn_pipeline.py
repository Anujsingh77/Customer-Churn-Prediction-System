"""
churn_pipeline.py
Full pipeline: data loading → EDA → feature engineering → model training
→ evaluation → feature importance → visualization outputs.
"""
import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
import pickle

warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ── Colour palette ──────────────────────────────────────────────────────────
PALETTE = {
    "primary":   "#5340C9",
    "secondary": "#378ADD",
    "danger":    "#E24B4A",
    "success":   "#3B6D11",
    "warning":   "#BA7517",
    "neutral":   "#888780",
    "bg":        "#F8F7FF",
    "churn_yes": "#E24B4A",
    "churn_no":  "#5340C9",
}
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.25,
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
})

# ═══════════════════════════════════════════════════════════════
# 1. LOAD & CLEAN
# ═══════════════════════════════════════════════════════════════
print("=" * 55)
print("  CUSTOMER CHURN PREDICTION PIPELINE")
print("=" * 55)

df = pd.read_csv("data/customers.csv")
print(f"\n[1/6] Data loaded  →  {df.shape[0]:,} rows × {df.shape[1]} cols")
print(f"      Churn rate   →  {df.churn.mean():.1%}")
print(f"      Nulls        →  {df.isnull().sum().sum()}")

# ═══════════════════════════════════════════════════════════════
# 2. EDA PLOTS
# ═══════════════════════════════════════════════════════════════
print("\n[2/6] Generating EDA visualisations…")

# ── EDA Figure 1: Overview ──────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("Customer Churn — Exploratory Data Analysis", fontsize=16, fontweight="bold", y=1.01)

# Churn distribution
ax = axes[0, 0]
counts = df["churn"].value_counts()
bars = ax.bar(["Retained", "Churned"], counts.values,
              color=[PALETTE["churn_no"], PALETTE["churn_yes"]], width=0.5)
ax.set_title("Churn distribution")
ax.set_ylabel("Customers")
for b in bars:
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 20,
            f"{b.get_height():,}\n({b.get_height()/len(df):.1%})",
            ha="center", va="bottom", fontsize=10)

# Tenure by churn
ax = axes[0, 1]
for label, color, name in [(0, PALETTE["churn_no"], "Retained"), (1, PALETTE["churn_yes"], "Churned")]:
    ax.hist(df[df.churn == label]["tenure"], bins=25, alpha=0.65,
            color=color, label=name, edgecolor="white")
ax.set_title("Tenure distribution")
ax.set_xlabel("Months")
ax.legend(frameon=False)

# Monthly charges by churn
ax = axes[0, 2]
for label, color, name in [(0, PALETTE["churn_no"], "Retained"), (1, PALETTE["churn_yes"], "Churned")]:
    ax.hist(df[df.churn == label]["monthly_charges"], bins=25, alpha=0.65,
            color=color, label=name, edgecolor="white")
ax.set_title("Monthly charges ($)")
ax.set_xlabel("$ / month")
ax.legend(frameon=False)

# Churn by contract type
ax = axes[1, 0]
ct = df.groupby("contract")["churn"].mean().sort_values(ascending=False)
bars = ax.barh(ct.index, ct.values * 100,
               color=[PALETTE["churn_yes"] if v > 0.3 else PALETTE["secondary"] for v in ct.values])
ax.set_title("Churn rate by contract")
ax.set_xlabel("Churn rate (%)")
for b in bars:
    ax.text(b.get_width() + 0.5, b.get_y() + b.get_height()/2,
            f"{b.get_width():.1f}%", va="center", fontsize=10)

# Churn by internet service
ax = axes[1, 1]
ct2 = df.groupby("internet_service")["churn"].mean().sort_values(ascending=False)
bars2 = ax.barh(ct2.index, ct2.values * 100,
                color=[PALETTE["churn_yes"] if v > 0.25 else PALETTE["secondary"] for v in ct2.values])
ax.set_title("Churn rate by internet service")
ax.set_xlabel("Churn rate (%)")
for b in bars2:
    ax.text(b.get_width() + 0.5, b.get_y() + b.get_height()/2,
            f"{b.get_width():.1f}%", va="center", fontsize=10)

# Scatter: tenure vs monthly charges
ax = axes[1, 2]
sample = df.sample(600, random_state=1)
ax.scatter(sample[sample.churn==0]["tenure"], sample[sample.churn==0]["monthly_charges"],
           alpha=0.35, s=18, color=PALETTE["churn_no"], label="Retained")
ax.scatter(sample[sample.churn==1]["tenure"], sample[sample.churn==1]["monthly_charges"],
           alpha=0.45, s=18, color=PALETTE["churn_yes"], label="Churned")
ax.set_title("Tenure vs monthly charges")
ax.set_xlabel("Tenure (months)")
ax.set_ylabel("Monthly charges ($)")
ax.legend(frameon=False, fontsize=9)

plt.tight_layout()
plt.savefig("outputs/eda_overview.png", dpi=150, bbox_inches="tight")
plt.close()
print("      → outputs/eda_overview.png")

# ── EDA Figure 2: Correlation heatmap ──────────────────────────
num_cols = ["tenure", "monthly_charges", "total_charges", "num_services",
            "senior_citizen", "partner", "dependents", "churn"]
fig, ax = plt.subplots(figsize=(9, 7))
corr = df[num_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(240, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap=cmap,
            center=0, square=True, linewidths=0.5, ax=ax,
            cbar_kws={"shrink": .8})
ax.set_title("Feature correlation matrix", pad=12)
plt.tight_layout()
plt.savefig("outputs/eda_correlation.png", dpi=150, bbox_inches="tight")
plt.close()
print("      → outputs/eda_correlation.png")

# ═══════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════
print("\n[3/6] Feature engineering…")

df_model = df.copy()

# Encode categoricals
le = LabelEncoder()
for col in ["internet_service", "contract", "payment_method"]:
    df_model[col + "_enc"] = le.fit_transform(df_model[col])

# New features
df_model["avg_monthly_spend"]    = df_model["total_charges"] / (df_model["tenure"] + 1)
df_model["charge_per_service"]   = df_model["monthly_charges"] / (df_model["num_services"] + 1)
df_model["is_new_customer"]      = (df_model["tenure"] <= 3).astype(int)
df_model["is_long_term"]         = (df_model["tenure"] >= 48).astype(int)
df_model["has_fiber"]            = (df_model["internet_service"] == "Fiber optic").astype(int)
df_model["month_to_month"]       = (df_model["contract"] == "Month-to-month").astype(int)
df_model["electronic_check"]     = (df_model["payment_method"] == "Electronic check").astype(int)

FEATURES = [
    "tenure", "monthly_charges", "total_charges", "num_services",
    "senior_citizen", "partner", "dependents", "phone_service",
    "multiple_lines", "online_security", "online_backup",
    "device_protection", "tech_support", "streaming_tv", "streaming_movies",
    "paperless_billing",
    "internet_service_enc", "contract_enc", "payment_method_enc",
    "avg_monthly_spend", "charge_per_service",
    "is_new_customer", "is_long_term", "has_fiber",
    "month_to_month", "electronic_check",
]

X = df_model[FEATURES]
y = df_model["churn"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"      Train: {len(X_train):,}  |  Test: {len(X_test):,}")
print(f"      Features: {len(FEATURES)}")

# ═══════════════════════════════════════════════════════════════
# 4. MODEL TRAINING & COMPARISON
# ═══════════════════════════════════════════════════════════════
print("\n[4/6] Training models…")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, C=0.5),
    "Random Forest":       RandomForestClassifier(n_estimators=200, max_depth=8,
                                                  min_samples_leaf=5, random_state=42, n_jobs=-1),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=150, max_depth=4,
                                                       learning_rate=0.08, random_state=42),
}

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

results = {}
for name, model in models.items():
    Xtr = X_train_s if name == "Logistic Regression" else X_train
    Xte = X_test_s  if name == "Logistic Regression" else X_test
    model.fit(Xtr, y_train)
    y_pred  = model.predict(Xte)
    y_proba = model.predict_proba(Xte)[:, 1]
    auc     = roc_auc_score(y_test, y_proba)
    ap      = average_precision_score(y_test, y_proba)
    cv_auc  = cross_val_score(model, Xtr, y_train,
                               cv=StratifiedKFold(5), scoring="roc_auc").mean()
    results[name] = {"model": model, "y_pred": y_pred, "y_proba": y_proba,
                     "auc": auc, "ap": ap, "cv_auc": cv_auc}
    print(f"      {name:<25}  AUC={auc:.4f}  AP={ap:.4f}  CV-AUC={cv_auc:.4f}")

# Best model
best_name = max(results, key=lambda k: results[k]["auc"])
best      = results[best_name]
print(f"\n      ★ Best model: {best_name}  (AUC={best['auc']:.4f})")

# Save
with open("models/best_model.pkl", "wb") as f:
    pickle.dump({"model": best["model"], "scaler": scaler,
                 "features": FEATURES, "name": best_name}, f)
print("      Model saved → models/best_model.pkl")

# ═══════════════════════════════════════════════════════════════
# 5. EVALUATION PLOTS
# ═══════════════════════════════════════════════════════════════
print("\n[5/6] Generating model evaluation plots…")

# ── Figure 3: Model comparison + ROC + confusion ───────────────
fig = plt.figure(figsize=(18, 11))
fig.suptitle("Model Evaluation Dashboard", fontsize=16, fontweight="bold")
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.35)

# Model comparison bar
ax0 = fig.add_subplot(gs[0, 0])
names = list(results.keys())
aucs  = [results[n]["auc"] for n in names]
colors = [PALETTE["primary"] if n == best_name else PALETTE["neutral"] for n in names]
bars  = ax0.barh(names, aucs, color=colors, height=0.45)
ax0.set_xlim(0.5, 1.0)
ax0.set_title("Model comparison (ROC-AUC)")
ax0.set_xlabel("AUC")
for b, v in zip(bars, aucs):
    ax0.text(v + 0.003, b.get_y() + b.get_height()/2,
             f"{v:.4f}", va="center", fontsize=10)

# ROC curves
ax1 = fig.add_subplot(gs[0, 1])
ax1.plot([0,1],[0,1],"--", color=PALETTE["neutral"], linewidth=1)
line_colors = [PALETTE["primary"], PALETTE["secondary"], PALETTE["warning"]]
for i, (name, res) in enumerate(results.items()):
    fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
    ax1.plot(fpr, tpr, color=line_colors[i], linewidth=2,
             label=f"{name} ({res['auc']:.3f})")
ax1.set_title("ROC curves")
ax1.set_xlabel("False positive rate")
ax1.set_ylabel("True positive rate")
ax1.legend(frameon=False, fontsize=9)

# Precision-Recall curve (best model)
ax2 = fig.add_subplot(gs[0, 2])
prec, rec, _ = precision_recall_curve(y_test, best["y_proba"])
ax2.plot(rec, prec, color=PALETTE["primary"], linewidth=2)
ax2.axhline(y_test.mean(), color=PALETTE["danger"], linestyle="--",
            linewidth=1, label=f"Baseline ({y_test.mean():.2f})")
ax2.set_title(f"Precision-Recall — {best_name}")
ax2.set_xlabel("Recall")
ax2.set_ylabel("Precision")
ax2.legend(frameon=False)

# Confusion matrix
ax3 = fig.add_subplot(gs[1, 0])
cm = confusion_matrix(y_test, best["y_pred"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax3,
            xticklabels=["Retained", "Churned"],
            yticklabels=["Retained", "Churned"],
            linewidths=0.5, cbar=False)
ax3.set_title(f"Confusion matrix — {best_name}")
ax3.set_xlabel("Predicted")
ax3.set_ylabel("Actual")

# Probability distribution
ax4 = fig.add_subplot(gs[1, 1])
ax4.hist(best["y_proba"][y_test == 0], bins=30, alpha=0.65,
         color=PALETTE["churn_no"], label="Retained", edgecolor="white")
ax4.hist(best["y_proba"][y_test == 1], bins=30, alpha=0.65,
         color=PALETTE["churn_yes"], label="Churned", edgecolor="white")
ax4.axvline(0.5, color="black", linestyle="--", linewidth=1)
ax4.set_title("Predicted churn probability")
ax4.set_xlabel("P(churn)")
ax4.legend(frameon=False)

# Classification report heatmap
ax5 = fig.add_subplot(gs[1, 2])
report = classification_report(y_test, best["y_pred"],
                                target_names=["Retained","Churned"],
                                output_dict=True)
report_df = pd.DataFrame(report).T.drop("accuracy").drop(columns=["support"])
sns.heatmap(report_df.astype(float), annot=True, fmt=".2f", cmap="Blues",
            ax=ax5, linewidths=0.5, cbar=False, vmin=0, vmax=1)
ax5.set_title("Classification report")

plt.savefig("outputs/model_evaluation.png", dpi=150, bbox_inches="tight")
plt.close()
print("      → outputs/model_evaluation.png")

# ── Figure 4: Feature importance ───────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle(f"Feature Importance — {best_name}", fontsize=15, fontweight="bold")

model_obj = best["model"]

# Built-in importance (RF/GB have feature_importances_)
if hasattr(model_obj, "feature_importances_"):
    imp = pd.Series(model_obj.feature_importances_, index=FEATURES).sort_values()
    top = imp.tail(18)
    colors_imp = [PALETTE["primary"] if v > top.median() else PALETTE["secondary"] for v in top.values]
    top.plot(kind="barh", ax=axes[0], color=colors_imp)
    axes[0].set_title("Built-in feature importance")
    axes[0].set_xlabel("Mean decrease in impurity")
elif hasattr(model_obj, "coef_"):
    coef = pd.Series(np.abs(model_obj.coef_[0]), index=FEATURES).sort_values()
    coef.tail(18).plot(kind="barh", ax=axes[0], color=PALETTE["primary"])
    axes[0].set_title("Logistic regression |coefficients|")
    axes[0].set_xlabel("|Coefficient|")

# Permutation importance
Xte_use = X_test_s if best_name == "Logistic Regression" else X_test
perm    = permutation_importance(model_obj, Xte_use, y_test,
                                  n_repeats=20, random_state=42, n_jobs=-1)
perm_df = pd.DataFrame({"feature": FEATURES,
                         "importance": perm.importances_mean,
                         "std": perm.importances_std}) \
            .sort_values("importance").tail(18)

colors_p = [PALETTE["danger"] if v > perm_df["importance"].median()
            else PALETTE["warning"] for v in perm_df["importance"]]
axes[1].barh(perm_df["feature"], perm_df["importance"],
             xerr=perm_df["std"], color=colors_p, capsize=3)
axes[1].set_title("Permutation importance (test set)")
axes[1].set_xlabel("Mean accuracy decrease")

plt.tight_layout()
plt.savefig("outputs/feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("      → outputs/feature_importance.png")

# ── Figure 5: Business insights ────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Business Insights & Segment Analysis", fontsize=15, fontweight="bold")

# Risk segments
df_score = df.copy()
Xall     = df_model[FEATURES]
Xall_use = scaler.transform(Xall) if best_name == "Logistic Regression" else Xall
df_score["churn_prob"] = model_obj.predict_proba(Xall_use)[:, 1]
bins   = [0, 0.3, 0.6, 1.0]
labels = ["Low risk", "Medium risk", "High risk"]
df_score["risk_segment"] = pd.cut(df_score["churn_prob"], bins=bins, labels=labels)

seg_counts = df_score["risk_segment"].value_counts().reindex(labels)
seg_colors = [PALETTE["success"], PALETTE["warning"], PALETTE["danger"]]
wedges, texts, autotexts = axes[0].pie(
    seg_counts, labels=labels, autopct="%1.1f%%",
    colors=seg_colors, startangle=90,
    wedgeprops={"edgecolor": "white", "linewidth": 1.5})
for at in autotexts:
    at.set_fontsize(10)
axes[0].set_title("Customer risk segments")

# Revenue at risk
df_score["monthly_revenue_risk"] = df_score["churn_prob"] * df_score["monthly_charges"]
seg_rev = df_score.groupby("risk_segment", observed=True)["monthly_revenue_risk"].sum()
bars = axes[1].bar(seg_rev.index, seg_rev.values / 1000, color=seg_colors, width=0.5)
axes[1].set_title("Monthly revenue at risk ($k)")
axes[1].set_ylabel("$k / month")
for b in bars:
    axes[1].text(b.get_x() + b.get_width()/2, b.get_height() + 0.2,
                 f"${b.get_height():.1f}k", ha="center", fontsize=10)

# Top 20 high-risk customers
top20 = df_score.nlargest(20, "churn_prob")[
    ["customer_id", "tenure", "monthly_charges", "churn_prob", "contract"]
].reset_index(drop=True)
sc = axes[2].scatter(top20["tenure"], top20["monthly_charges"],
               c=top20["churn_prob"], cmap="RdYlGn_r",
               s=120, edgecolors="white", linewidth=0.8, vmin=0.5, vmax=1.0)
plt.colorbar(sc, ax=axes[2], label="Churn probability")
axes[2].set_title("Top 20 highest-risk customers")
axes[2].set_xlabel("Tenure (months)")
axes[2].set_ylabel("Monthly charges ($)")

plt.tight_layout()
plt.savefig("outputs/business_insights.png", dpi=150, bbox_inches="tight")
plt.close()
print("      → outputs/business_insights.png")

# ── Save scored dataset ─────────────────────────────────────────
df_score.to_csv("outputs/customers_scored.csv", index=False)
print("      → outputs/customers_scored.csv")

# ═══════════════════════════════════════════════════════════════
# 6. SUMMARY REPORT
# ═══════════════════════════════════════════════════════════════
print("\n[6/6] Summary")
print("─" * 55)
print(f"  Dataset:         {len(df):,} customers")
print(f"  Churn rate:      {df.churn.mean():.1%}")
print(f"  Best model:      {best_name}")
print(f"  ROC-AUC:         {best['auc']:.4f}")
print(f"  Avg Precision:   {best['ap']:.4f}")
total_rev_risk = df_score["monthly_revenue_risk"].sum()
high_risk_n    = (df_score["risk_segment"] == "High risk").sum()
print(f"  High-risk custos:{high_risk_n:,}")
print(f"  Monthly rev risk:${total_rev_risk:,.0f}")
print("─" * 55)
print("\n  Outputs saved to:  outputs/")
print("  Model saved to:    models/best_model.pkl")
print("\nDone! ✓\n")
