"""
generate_data.py
Generates a realistic synthetic telecom customer churn dataset.
"""
import numpy as np
import pandas as pd

np.random.seed(42)
N = 2000

def generate_churn_data(n=N):
    tenure       = np.random.randint(1, 73, n)
    monthly_charges = np.round(np.random.uniform(18, 120, n), 2)
    total_charges   = np.round(monthly_charges * tenure * np.random.uniform(0.85, 1.05, n), 2)

    contract = np.random.choice(["Month-to-month", "One year", "Two year"],
                                 n, p=[0.55, 0.25, 0.20])
    payment  = np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        n, p=[0.34, 0.22, 0.22, 0.22])
    internet = np.random.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22])

    senior      = np.random.choice([0, 1], n, p=[0.84, 0.16])
    partner     = np.random.choice([0, 1], n, p=[0.50, 0.50])
    dependents  = np.random.choice([0, 1], n, p=[0.70, 0.30])
    phone       = np.random.choice([0, 1], n, p=[0.10, 0.90])
    multiple    = np.where(phone == 1, np.random.choice([0, 1], n, p=[0.53, 0.47]), 0)
    online_sec  = np.random.choice([0, 1], n, p=[0.50, 0.50])
    online_bkp  = np.random.choice([0, 1], n, p=[0.56, 0.44])
    device_prot = np.random.choice([0, 1], n, p=[0.56, 0.44])
    tech_support= np.random.choice([0, 1], n, p=[0.50, 0.50])
    streaming_tv= np.random.choice([0, 1], n, p=[0.60, 0.40])
    streaming_mv= np.random.choice([0, 1], n, p=[0.60, 0.40])
    num_services= online_sec + online_bkp + device_prot + tech_support + streaming_tv + streaming_mv + multiple

    # Churn probability – engineered to mimic real patterns
    logit  =  2.5
    logit += -0.06 * tenure
    logit += -0.02 * monthly_charges
    logit +=  0.8  * (contract == "Month-to-month")
    logit += -0.9  * (contract == "Two year")
    logit +=  0.5  * (payment == "Electronic check")
    logit +=  0.6  * (internet == "Fiber optic")
    logit += -0.3  * (internet == "No")
    logit +=  0.3  * senior
    logit += -0.2  * partner
    logit += -0.1  * num_services
    prob   = 1 / (1 + np.exp(-logit))
    churn  = (np.random.rand(n) < prob).astype(int)

    customer_id = [f"CUST{str(i).zfill(5)}" for i in range(1, n+1)]

    df = pd.DataFrame({
        "customer_id":     customer_id,
        "tenure":          tenure,
        "monthly_charges": monthly_charges,
        "total_charges":   total_charges,
        "senior_citizen":  senior,
        "partner":         partner,
        "dependents":      dependents,
        "phone_service":   phone,
        "multiple_lines":  multiple,
        "internet_service":internet,
        "online_security": online_sec,
        "online_backup":   online_bkp,
        "device_protection":device_prot,
        "tech_support":    tech_support,
        "streaming_tv":    streaming_tv,
        "streaming_movies":streaming_mv,
        "contract":        contract,
        "paperless_billing":np.random.choice([0,1], n, p=[0.40,0.60]),
        "payment_method":  payment,
        "num_services":    num_services,
        "churn":           churn,
    })
    return df

if __name__ == "__main__":
    df = generate_churn_data()
    df.to_csv("data/customers.csv", index=False)
    print(f"Dataset saved: {len(df)} rows, churn rate = {df.churn.mean():.1%}")
