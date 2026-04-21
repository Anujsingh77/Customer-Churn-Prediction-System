# Customer Churn Prediction System
### Full ML Pipeline + Interactive Dashboard

---

## Project Structure

```
churn_predictor/
├── data/
│   ├── generate_data.py     # Synthetic dataset generator
│   └── customers.csv        # Generated after step 2
├── models/
│   └── best_model.pkl       # Saved model after training
├── outputs/
│   ├── customers_scored.csv
│   ├── eda_overview.png
│   ├── eda_correlation.png
│   ├── model_evaluation.png
│   ├── feature_importance.png
│   └── business_insights.png
├── churn_pipeline.py        # Full ML pipeline
├── dashboard.py             # Streamlit dashboard
└── requirements.txt
```

---

## Step-by-Step Guide

### Step 1 — Install Python
Make sure you have Python 3.9+ installed.
Download from: https://www.python.org/downloads/

Verify:
```
python --version
```

---

### Step 2 — Create a virtual environment (recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

---

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

This installs: pandas, numpy, matplotlib, seaborn, scikit-learn, streamlit.

---

### Step 4 — Generate the dataset
```bash
cd data
python generate_data.py
cd ..
```

Output: `data/customers.csv` with 2,000 synthetic telecom customers.

The dataset includes:
- Demographics (senior, partner, dependents)
- Services (internet, phone, streaming, security...)
- Account info (tenure, contract, payment method, charges)
- Target variable: `churn` (0 = retained, 1 = churned)

---

### Step 5 — Run the full ML pipeline
```bash
python churn_pipeline.py
```

This runs 6 stages:
1. **Load & clean** — reads CSV, checks nulls
2. **EDA** — generates 2 visualisation PNGs
3. **Feature engineering** — creates 26 features including engineered ones
4. **Model training** — trains and compares 3 models:
   - Logistic Regression
   - Random Forest
   - Gradient Boosting
5. **Evaluation plots** — ROC curves, confusion matrix, probability distributions
6. **Feature importance** — built-in + permutation importance

All outputs are saved to the `outputs/` folder.
Best model is saved to `models/best_model.pkl`.

---

### Step 6 — Launch the dashboard
```bash
streamlit run dashboard.py
```

Open your browser at: http://localhost:8501

## DEMO: https://anujsingh77-customer-churn-prediction-system-dashboard-kjllqy.streamlit.app/

The dashboard has 4 tabs:

| Tab | What you see |
|-----|-------------|
| Overview | KPIs, churn distribution, scatter plots |
| Risk Segments | Breakdown by High / Medium / Low risk |
| Predict Customer | Enter a new customer, get instant prediction |
| Customer List | Filterable, sortable table with CSV export |

---

## Using Your Own Data

Replace `data/customers.csv` with your own CSV. Make sure your file has:

| Column | Type | Description |
|--------|------|-------------|
| tenure | int | Months as customer |
| monthly_charges | float | Monthly bill |
| total_charges | float | Total billed |
| contract | str | Month-to-month / One year / Two year |
| internet_service | str | DSL / Fiber optic / No |
| payment_method | str | Electronic check / Mailed check / etc |
| churn | int | 0 = retained, 1 = churned |
| (other features) | int | 0/1 binary flags |

Then re-run `churn_pipeline.py` to retrain on your data.

---

## Model Explanation

### Features Used (26 total)
- **Raw**: tenure, monthly_charges, total_charges, num_services, demographics, services
- **Engineered**:
  - `avg_monthly_spend` = total_charges / (tenure + 1)
  - `charge_per_service` = monthly_charges / (num_services + 1)
  - `is_new_customer` = tenure ≤ 3 months
  - `is_long_term` = tenure ≥ 48 months
  - `has_fiber`, `month_to_month`, `electronic_check` (binary flags)

### Why these features matter
| Feature | Impact |
|---------|--------|
| Contract type | Month-to-month → highest churn |
| Tenure | Shorter = higher risk |
| Fiber optic | Correlated with higher churn (premium pricing) |
| Electronic check | Highest-risk payment method |
| Num services | More services = more retention |

### Scoring thresholds
- **High risk**: P(churn) ≥ 0.60
- **Medium risk**: P(churn) 0.30 – 0.59
- **Low risk**: P(churn) < 0.30

---

## Typical Results
- ROC-AUC: ~0.85–0.90
- Churn rate in synthetic data: ~28–32%
- Top predictors: contract type, tenure, monthly charges

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| Dashboard shows no data | Run `churn_pipeline.py` first |
| Port 8501 in use | `streamlit run dashboard.py --server.port 8502` |
| Slow training | Reduce `n_estimators` in `churn_pipeline.py` |

## Some Prediction Screenshots:
![alt text](../business_insights.png)
![alt text](../eda_overview.png)
![alt text](../feature_importance.png)
![alt text](../model_evaluation.png)

## NOTE:
If you don't see the screenshots so you need to download the README.md file then opne with vs_code then you see the screenshots.
