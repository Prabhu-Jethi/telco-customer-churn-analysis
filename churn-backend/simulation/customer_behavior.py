from pathlib import Path
import pandas as pd
import numpy as np

RANDOM_SEED = 42
DAYS = 90

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (BASE_DIR / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv")
OUTPUT_FILE = (BASE_DIR / "data" / "customer_logs.csv")


def generate_behavior_logs(customers: pd.DataFrame):
    rng = np.random.default_rng(RANDOM_SEED)
    records = []

    for _, customer in customers.iterrows():
        customer_id = customer["customerID"]
        tenure = float(customer["tenure"])
        monthly_charges = float(customer["MonthlyCharges"])
        contract = customer["Contract"]
        internet = customer["InternetService"]

        engagement_score = rng.normal(1.0, 0.15)
        is_churner = (customer["Churn"] == "Yes")

        if contract == "Month-to-month":
            engagement_score *= 0.95
        elif contract == "Two year":
            engagement_score *= 1.05

        if internet == "Fiber optic":
            engagement_score *= 1.02
            
        # Add realistic overlap: Only 60% of churners show reduced engagement
        # and 10% of non-churners randomly have low engagement.
        shows_churn_behavior = is_churner and rng.random() < 0.60
        shows_vacation_behavior = not is_churner and rng.random() < 0.10

        if shows_churn_behavior or shows_vacation_behavior:
            engagement_score *= rng.uniform(0.5, 0.8) 

        base_login = max(1, 12 + (tenure * 0.03) + rng.normal(0, 1.5))
        base_usage = max(2, 25 + (monthly_charges * 0.12) + rng.normal(0, 3))

        for day in range(DAYS):
            if shows_churn_behavior:
                # Gradual drop off, but with more noise
                drop_factor = (day / DAYS) ** 1.5 
                trend = max(0.1, 1 - drop_factor * rng.uniform(0.3, 0.7))
            elif shows_vacation_behavior:
                # Random mid-period dip
                trend = max(0.2, 1 - rng.uniform(0.2, 0.5))
            else:
                # Normal users stay relatively stable
                trend = 1 - (day / DAYS) * rng.uniform(0.00, 0.12)

            daily_noise = rng.normal(1.0, 0.12)

            login_count = max(0, int(base_login * engagement_score * trend * daily_noise))

            feature_usage = max(0, int(base_usage * engagement_score * trend * daily_noise))

            session_minutes = max(0, round(feature_usage * rng.uniform(2.0, 5.0), 2))

            active = int(login_count > 0)

            ticket_probability = (0.03 if active else 0.08)

            support_ticket = int(rng.random() < ticket_probability)

            resolution_hours = (
                round(rng.uniform(2, 72), 2) if support_ticket else 0.0
            )

            records.append({
                "customer_id": customer_id,
                "date": pd.Timestamp("2025-01-01")
                + pd.Timedelta(days=day),
                "login_count": login_count,
                "session_minutes": session_minutes,
                "feature_usage_count": feature_usage,
                "active": active,
                "support_tickets": support_ticket,
                "resolution_hours": resolution_hours,
            })

    return pd.DataFrame(records)


def main():
    customers = pd.read_csv(INPUT_FILE)
    behavior_logs = generate_behavior_logs(customers)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    behavior_logs.to_csv(OUTPUT_FILE, index=False)
    
    print(f"Created: {OUTPUT_FILE}")
    print(f"Rows: {len(behavior_logs):,}")
    print(f"Customers: {behavior_logs['customer_id'].nunique(): ,}")


if __name__ == "__main__":
    main()