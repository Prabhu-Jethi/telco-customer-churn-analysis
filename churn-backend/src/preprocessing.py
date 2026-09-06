import os
import joblib
import pandas as pd

## Paths 

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(BASE_DIR, "models")

FEATURE_COLUMNS_PATH = os.path.join(
    MODEL_DIR,
    "feature_columns.pkl"
)

SCALER_PATH = os.path.join(
    MODEL_DIR,
    "scaler.pkl"
)

## Preprocessed models

feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
scaler = joblib.load(SCALER_PATH)


BINARY_COLUMNS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "PaperlessBilling",
]

ONE_HOT_COLUMNS = [
    "Contract",
    "PaymentMethod",
    "InternetService",
]

BINARY_MAPPINGS = {
    "gender": {
        "Female": 0,
        "Male": 1,
    },

    "Partner": {
        "No": 0,
        "Yes": 1,
    },

    "Dependents": {
        "No": 0,
        "Yes": 1,
    },

    "PhoneService": {
        "No": 0,
        "Yes": 1,
    },

    "MultipleLines": {
        "No": 0,
        "No phone service": 1,
        "Yes": 2,
    },

    "OnlineSecurity": {
        "No": 0,
        "No internet service": 1,
        "Yes": 2,
    },

    "OnlineBackup": {
        "No": 0,
        "No internet service": 1,
        "Yes": 2,
    },

    "DeviceProtection": {
        "No": 0,
        "No internet service": 1,
        "Yes": 2,
    },

    "TechSupport": {
        "No": 0,
        "No internet service": 1,
        "Yes": 2,
    },

    "StreamingTV": {
        "No": 0,
        "No internet service": 1,
        "Yes": 2,
    },

    "StreamingMovies": {
        "No": 0,
        "No internet service": 1,
        "Yes": 2,
    },

    "PaperlessBilling": {
        "No": 0,
        "Yes": 1,
    },
}


def preprocess_input(customer_data):
    """
    Convert raw customer data into the exact feature
    representation expected by the trained model.

    Parameters
    ----------
    customer_data : dict
        Raw customer data received from FastAPI.

    Returns
    -------
    pd.DataFrame
        Scaled 26-feature model input.
    """

    # 1. Convert JSON/dict into DataFrame

    df = pd.DataFrame([customer_data])


    # 2. Handle TotalCharges

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"],
            errors="coerce"
        )


    # 3. Encode binary columns

    for column in BINARY_COLUMNS:

        if column not in df.columns:
            raise ValueError(
                f"Missing required column: {column}"
            )

        mapping = BINARY_MAPPINGS[column]

        df[column] = df[column].map(mapping)

        if df[column].isnull().any():
            raise ValueError(
                f"Invalid value found in column '{column}'"
            )


    # 4. One-hot encode categorical columns

    df = pd.get_dummies(
        df,
        columns=ONE_HOT_COLUMNS,
        dtype=int
    )


    # 5. Remove columns not used by the model

    df = df.drop(
        columns=["customerID", "Churn"],
        errors="ignore"
    )

    
    # 6. Align with training feature columns

    df = df.reindex(
        columns=feature_columns,
        fill_value=0
    )


    # 7. Apply the SAME scaler used during training

    X_scaled = scaler.transform(df)


    # 8. Return model-ready data

    return pd.DataFrame(
        X_scaled,
        columns=feature_columns
    )