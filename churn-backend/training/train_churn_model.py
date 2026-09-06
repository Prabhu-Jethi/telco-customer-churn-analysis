from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
MASTER_FILE = DATA_DIR / "master_training_data.csv"

def load_and_preprocess(filepath):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Fix TotalCharges
    df['TotalCharges'] = df['TotalCharges'].replace(' ', float('nan'))
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(subset=['TotalCharges'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # Encode target
    df['Churn'] = (df['Churn'] == 'Yes').astype(int)
    
    # Encode binary columns
    bin_cols = ['gender','Partner','Dependents','PhoneService',
                'MultipleLines','OnlineSecurity','OnlineBackup',
                'DeviceProtection','TechSupport','StreamingTV',
                'StreamingMovies','PaperlessBilling']
    le = LabelEncoder()
    for col in bin_cols:
        df[col] = le.fit_transform(df[col])
    
    # One-hot encode
    df = pd.get_dummies(df, columns=['Contract','PaymentMethod','InternetService'])
    
    X = df.drop(['customerID','Churn'], axis=1)
    y = df['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    # SMOTE on train only
    print("Applying SMOTE...")
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    
    # Scale
    print("Scaling features...")
    scaler = StandardScaler()
    X_res = scaler.fit_transform(X_res)
    X_test = scaler.transform(X_test)
    
    return X_res, X_test, y_res, y_test, scaler, X.columns.to_list()


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    X_res, X_test, y_res, y_test, scaler, feature_columns = load_and_preprocess(MASTER_FILE)

    models = {
      'Logistic Regression': LogisticRegression(max_iter=1000),
      'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
      'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }

    print("\nTraining models...")
    for name, model in models.items():
        model.fit(X_res, y_res)
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:,1]
        print(f"\n{name}")
        print(classification_report(y_test, preds))
        print(f"ROC-AUC: {roc_auc_score(y_test, proba):.3f}")

    # Save the BEST performing model (Random Forest)
    print("\nSaving Random Forest model to disk...")
    best_model = models['Random Forest']
    
    joblib.dump(best_model, MODELS_DIR / 'model.pkl')
    joblib.dump(scaler, MODELS_DIR / 'scaler.pkl')
    joblib.dump(feature_columns, MODELS_DIR / 'feature_columns.pkl')

    print(f"Models successfully saved to {MODELS_DIR}!")

if __name__ == "__main__":
    main()
