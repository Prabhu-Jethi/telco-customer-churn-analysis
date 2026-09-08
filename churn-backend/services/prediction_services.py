import sys
import joblib
import shap
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
from src.preprocessing import preprocess_input
from services.feature_service import get_customer_features

MODEL_PATH = BASE_DIR/"models"/"model.pkl"
SCALER_PATH = BASE_DIR/"models"/"scaler.pkl"
FEATURES_PATH = BASE_DIR/"models"/"feature_columns.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_columns = joblib.load(FEATURES_PATH)
explainer = shap.TreeExplainer(model)

### Function to format shap features - to display user friendly texts
def format_shap_feature(feature, impact):
    '''Convert internal model feature names into human-readable format.'''
    display_name = feature
    feature_group = feature

    if feature.startswith("Contract_"):
        contract = feature.replace("Contract_", "")
        display_name = f"{contract} contract"
        feature_group = "Contract"
    
    elif feature.startswith("PaymentMethod_"):
        payment = feature.replace("PaymentMethod_", "")
        display_name = f"{payment} payment"
        feature_group = "Payment Method"

    elif feature.startswith("InternetService_"):
        service = feature.replace("InternetService_", "")
        display_name = f"{service} internet service"
        feature_group = "Internet Service"

    else:
        display_names = {
            "gender": "Gender",
            "SeniorCitizen": "Senior citizen",
            "Partner": "Partner",
            "Dependents": "Dependents",
            "tenure": "Customer tenure",
            "PhoneService": "Phone service",
            "MultipleLines": "Multiple lines",
            "OnlineSecurity": "Online security",
            "OnlineBackup": "Online backup",
            "DeviceProtection": "Device protection",
            "TechSupport": "Technical support",
            "StreamingTV": "Streaming TV",
            "StreamingMovies": "Streaming movies",
            "PaperlessBilling": "Paperless billing",
            "MonthlyCharges": "Monthly charges",
            "TotalCharges": "Total charges",
            # Behavioral Features
            "login_count_30d": "Logins (30 days)",
            "session_minutes_30d": "Session minutes (30 days)",
            "feature_usage_30d": "Feature usage (30 days)",
            "support_tickets_30d": "Support tickets (30 days)",
            "resolution_hours_30d": "Ticket resolution hours (30 days)",
            "active_days_30d": "Active days (30 days)",
            "usage_change_30d": "Usage trend (%)",
            "login_count_90d": "Logins (90 days)",
            "session_minutes_90d": "Session minutes (90 days)",
            "feature_usage_90d": "Feature usage (90 days)",
            "support_tickets_90d": "Support tickets (90 days)",
            "resolution_hours_90d": "Ticket resolution hours (90 days)",
            "active_days_90d": "Active days (90 days)"
        }

        display_name = display_names.get(
            feature, feature
        )
        feature_group = display_name

    if impact > 0:
        direction = "increases_risk"
    else:
        direction = "reduces_risk"

    absolute_impact = abs(impact)
    if absolute_impact >= 0.50:
        importance = "High"
        
    elif absolute_impact >= 0.20:
        importance = "Medium"
        
    else:
        importance = "Low"

    return{
        "feature": feature_group,
        "display_name": display_name,
        "impact": round(float(impact), 4),
        "direction": direction,
        "importance": importance
    }
    
def predict_churn(customer_data):
    # Fetch behavioral data and merge it in!
    customer_id = customer_data.get("customerID")
    behavioral_features = None
    
    if customer_id:
        behavioral_features = get_customer_features(customer_id)
        
    # If the frontend sent a dummy/missing customer, simulate features based on their demographics!
    # This ensures the interactive UI sliders still produce logical high/low risk predictions.
    if not behavioral_features:
        tenure = int(customer_data.get("tenure", 0))
        contract = customer_data.get("Contract", "Month-to-month")
        
        # Create a 3-tier risk system for a balanced frontend experience
        if contract == "Month-to-month" and tenure <= 12:
            risk_tier = "high"
        elif contract == "Month-to-month" or (contract == "One year" and tenure < 24):
            risk_tier = "medium"
        else:
            risk_tier = "low"
            
        if risk_tier == "high":
            behavioral_features = {
                "login_count_30d": 2, "session_minutes_30d": 25, "feature_usage_30d": 2,
                "support_tickets_30d": 7, "resolution_hours_30d": 96, "active_days_30d": 2, "usage_change_30d": -85.0,
                "login_count_90d": 15, "session_minutes_90d": 180, "feature_usage_90d": 10,
                "support_tickets_90d": 12, "resolution_hours_90d": 160, "active_days_90d": 8
            }
        elif risk_tier == "medium":
            behavioral_features = {
                "login_count_30d": 12, "session_minutes_30d": 180, "feature_usage_30d": 15,
                "support_tickets_30d": 2, "resolution_hours_30d": 24, "active_days_30d": 14, "usage_change_30d": -15.0,
                "login_count_90d": 40, "session_minutes_90d": 600, "feature_usage_90d": 50,
                "support_tickets_90d": 4, "resolution_hours_90d": 48, "active_days_90d": 40
            }
        else:
            behavioral_features = {
                "login_count_30d": 22, "session_minutes_30d": 450, "feature_usage_30d": 40,
                "support_tickets_30d": 0, "resolution_hours_30d": 0, "active_days_30d": 24, "usage_change_30d": 2.0,
                "login_count_90d": 65, "session_minutes_90d": 1350, "feature_usage_90d": 120,
                "support_tickets_90d": 1, "resolution_hours_90d": 12, "active_days_90d": 70
            }
        
    behavioral_features.pop("customerID", None)
    customer_data.update(behavioral_features)
            
    X = preprocess_input(customer_data)

    probability = float(
        model.predict_proba(X.values)[0][1]
    )

    prediction = int(
        model.predict(X.values)[0]
    )

    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        values = shap_values[1][0]
    elif len(getattr(shap_values, 'shape', [])) == 3:
        values = shap_values[0, :, 1]
    else:
        values = shap_values[0]
    
    explanations = []

    for feature, shap_value in zip(
        feature_columns,
        values
    ):
        formatted = format_shap_feature(
            feature,
            float(shap_value)
        )

        explanations.append(formatted)

    ## sort by absolute SHAP impact
    explanations.sort(
        key=lambda x: abs(x["impact"]),
        reverse=True
    )

    return prediction, probability, explanations[:5]