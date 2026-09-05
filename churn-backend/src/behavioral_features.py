import pandas as pd
import os

def create_behavioral_features(behavior: pd.DataFrame):

    behavior = behavior.copy()
    behavior["date"] = pd.to_datetime(behavior["date"])
    latest_date = behavior["date"].max()
    recent_30 = behavior[
        behavior["date"] > latest_date - pd.Timedelta(days=30)
    ]
    recent_90 = behavior[
        behavior["date"] > latest_date - pd.Timedelta(days=90)
    ]

    features_30 = (recent_30.groupby("customer_id")
                   .agg(
                       avg_login_count_30=("login_count", "mean"),
                       active_days_30=("active", "sum"),
                       avg_feature_usage_30=("feature_usage_count", "mean"),
                       avg_session_minutes_30=("session_minutes", "mean"),
                       support_tickets_count_30=("support_tickets", "sum"),
                       avg_resolution_hours_30=("resolution_hours", "mean")
                   )
                   .reset_index()
        )
    
    features_90 = (recent_90.groupby("customer_id")
                   .agg(
                       avg_login_count_90=("login_count", "mean"),
                       active_days_90=("active", "sum"),
                       avg_feature_usage_90=("feature_usage_count", "mean")
                   )
                   .reset_index()
        )
    
    features = features_30.merge(features_90, on="customer_id", how="left")

    features["usage_change_30d"] = ((features["avg_feature_usage_30"] - features["avg_feature_usage_90"])
        / features["avg_feature_usage_90"].replace(0, 1)
    )

    features["login_change_30d"] = ((features["avg_login_count_30"] - features["avg_login_count_90"])
        / features["avg_login_count_90"].replace(0, 1)
    )

    last_activity = (
        behavior[behavior["active"] == 1]
        .groupby("customer_id")["date"]
        .max()
        .reset_index(name="last_activity_date")
    )

    features = features.merge(
        last_activity,
        on="customer_id",
        how="left",
    )

    features["days_since_last_activity"] = (
        latest_date
        - features["last_activity_date"]
    ).dt.days

    features = features.drop(columns=["last_activity_date"])

    return features


if __name__ == "__main__":

    input_file = "../data/customer_behavior_logs.csv"
    if os.path.exists(input_file):
        print(f"Reading {input_file}...")
        behavior_logs = pd.read_csv(input_file)
        
        print("Generating behavioral features...")
        behavioral_features = create_behavioral_features(behavior_logs)
        
        print("Merging features back to the original logs...")
        merged_logs = behavior_logs.merge(behavioral_features, on="customer_id", how="left")
        
        print(f"Saving merged data to {input_file}...")
        merged_logs.to_csv(input_file, index=False)
        print("Done!")
    else:
        print(f"File not found: {input_file}")