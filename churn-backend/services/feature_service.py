import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

PROFILE_FILE = DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
BEHAVIOR_FILE = DATA_DIR / "behavioral_features.csv"
MASTER_FILE = DATA_DIR / "master_training_data.csv"

def build_master_features():
    profiles = pd.read_csv(PROFILE_FILE)
    behavior = pd.read_csv(BEHAVIOR_FILE)
    behavior = behavior.rename(columns={"customer_id": "customerID"})
    
    # Left join so we keep all customers, even if they have no behavioral logs
    master_df = profiles.merge(behavior, on="customerID", how="left")
    
    # Fill missing behavioral features with 0 for customers who have no logs
    behavior_cols = behavior.columns.drop("customerID")
    master_df[behavior_cols] = master_df[behavior_cols].fillna(0)

    master_df.to_csv(MASTER_FILE, index=False)
    print(f"Master dataset created successfully with shape: {master_df.shape}")
    
    return master_df

def get_customer_features(customer_id: str):
    master_df = pd.read_csv(MASTER_FILE)
    customer_data = master_df[master_df["customerID"] == customer_id]
    
    if customer_data.empty:
        return None
    return customer_data.iloc[0].to_dict()

if __name__ == "__main__":
    build_master_features()
