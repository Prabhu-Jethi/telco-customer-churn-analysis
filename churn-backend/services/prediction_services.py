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
    if customer_id:
        behavioral_features = get_customer_features(customer_id)
        if behavioral_features:
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