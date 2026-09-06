# ChurnIQ — Customer Churn Prediction & Retention Intelligence Platform

> An end-to-end, explainable customer churn intelligence platform that predicts customer churn risk, explains the key drivers behind each prediction, and converts model outputs into actionable, behavior-driven retention recommendations.

**Live Application:** https://churniq-analysis-next.onrender.com
<br><br/>


---
<img width="1142" height="575" alt="Screenshot 2026-08-13 191639" src="https://github.com/user-attachments/assets/156248ec-5785-4f17-9c1c-7be195afd966" />
<img width="1047" height="373" alt="Screenshot 2026-08-13 191809" src="https://github.com/user-attachments/assets/99d44d62-6dee-4ab0-8157-b504c6475838" />
<img width="894" height="546" alt="Screenshot 2026-08-13 191839" src="https://github.com/user-attachments/assets/1e53ed87-dbab-4bd2-b8c3-1b585718eea9" />




## 🚀 Why ChurnIQ?

Customer churn is not just a machine-learning problem. A business needs to answer three questions:

1. **Which customers are likely to churn?**
2. **Why are they likely to churn?**
3. **What should the business do about it?**

A traditional churn model may provide only a probability score. That is often insufficient for business users because a score does not explain the reason behind the prediction or suggest an appropriate response.

**ChurnIQ connects all three stages into a single workflow:**

```text
Customer Data + Behavioral Logs
              │
              ▼
        Preprocessing
              │
              ▼
 Random Forest Churn Prediction
              │
              ▼
     SHAP Explainability
              │
              ▼
  Retention Recommendation
              │
              ▼
 Interactive ChurnIQ Interface
```

The result is a decision-support application rather than just a machine-learning model.

---

# 🎯 Business Problem

Telecom companies lose customers for many different reasons, including contract type, tenure, pricing, service configuration, and recent customer support experience.

The objective of this project is to identify customers with a high probability of churn early enough for a business to take preventive action.

The model is trained using the IBM Telco Customer Churn dataset combined with **simulated 90-day behavioral logs** (e.g., login frequency, support tickets, usage drop-offs) across **7,043 customers**. 

The project prioritizes **recall** because missing an actual churner can be more costly to the business than contacting a customer who ultimately stays.

---

# 💡 Solution

ChurnIQ provides an end-to-end ML workflow:

### 1. Predict

The system accepts a customer demographic profile, automatically queries the database for their latest **behavioral telemetry**, and sends the combined 39-feature payload to a FastAPI backend. 

The backend preprocesses the data and generates a churn probability using a highly-optimized **Random Forest** model.

### 2. Explain

SHAP is used to identify the most influential features contributing to an individual prediction.

Instead of returning:

```text
Churn probability: 76%
```

the system can explain:

```text
Usage dropped by 15% in 30 days → increases risk
Filed 4 support tickets           → increases risk
Month-to-month contract           → increases risk
```

### 3. Act

The strongest risk signals are converted into behavior-driven retention recommendations. If a customer is churning due to high support tickets, the system recommends white-glove technical support rather than a blanket monetary discount.

---

# 🧠 Machine Learning Pipeline

```text
  Demographics + Behavioral Logs
              │
              ▼
        Data Cleaning
              │
              ├── Handle missing TotalCharges
              ├── Convert numeric fields
              └── Encode categorical features
              │
              ▼
      Feature Engineering
              │
              ├── Label Encoding
              └── One-Hot Encoding
              │
              ▼
       Train/Test Split
              │
              ▼
            SMOTE
              │
              ▼
        StandardScaler
              │
              ▼
       Model Comparison
              │
              ├── Logistic Regression
              ├── Random Forest
              └── XGBoost
              │
              ▼
   Best Model: Random Forest
              │
              ▼
      SHAP Tree Explainer
              │
              ▼
   Prediction + Explanation
              │
              ▼
   Retention Recommendation
```

SMOTE is applied to the training data to address the class imbalance, while scaling is fitted on training data and then applied to the test data. The project compares Logistic Regression, Random Forest and XGBoost before selecting Random Forest as the strongest model (ROC-AUC: 0.923).

---

# 🏆 Model Performance

| Model               | Precision |   Recall | F1-Score |  ROC-AUC |
| ------------------- | --------: | -------: | -------: | -------: |
| XGBoost             |      0.79 |     0.71 |     0.75 |     0.911|
| Logistic Regression |      0.79 |     0.70 |     0.74 |     0.919|
| **Random Forest**   |  **0.81** | **0.70** | **0.75** | **0.923**|

The Random Forest model achieved the strongest combination of Precision and ROC-AUC (**0.923**) after the inclusion of the 13 new behavioral features, making it the selected production model.

---

# 🔍 Explainable AI with SHAP

ChurnIQ uses **SHAP (SHapley Additive exPlanations)** to explain individual predictions.

The explanation layer allows users to understand:

* Which features increased churn risk
* Which features reduced churn risk
* The relative impact of each feature
* Why a particular customer received their risk score

Important churn signals identified in the project include:

* **Behavioral Activity**: Usage drop-offs, platform abandonment (active days), and login frequency.
* **Friction Metrics**: The number of support tickets filed in the last 30/90 days and resolution times.
* **Financial/Contractual**: Contract type, tenure, and monthly charges.

---

# 🖥️ ChurnIQ Frontend

The frontend is built as a modern interactive web application rather than a notebook or static model demonstration.

### Main capabilities

* Customer risk profile
* Dynamic customer inputs
* Churn probability visualization
* Low / Medium / High risk classification
* SHAP driver visualization
* Behavior-driven retention recommendations
* Prediction workspace
* Explainability view
* Analytics view
* Prediction history interface
* Responsive UI
* API-driven prediction workflow

The frontend sends customer information to the FastAPI backend and renders the actual backend prediction rather than requiring the ML model to run directly inside the browser.

---

# ⚙️ Backend API

The FastAPI backend exposes a REST API for model inference.

### Endpoint

```http
POST /predict
```

### Example request

```json
{
  "customerID": "7043-ABCD",
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "No",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "Yes",
  "PaperlessBilling": "Yes",
  "MonthlyCharges": 75,
  "TotalCharges": 900,
  "Contract": "Month-to-month",
  "PaymentMethod": "Electronic check",
  "InternetService": "Fiber optic"
}
```

### Example response

```json
{
  "prediction": 1,
  "churn_probability": 0.8234,
  "churn_percentage": 82.34,
  "risk_level": "high",
  "model": "Random Forest",
  "drivers": [
    {
      "feature": "Usage trend (%)",
      "display_name": "Usage trend (%)",
      "impact": 0.1543,
      "direction": "increases_risk",
      "importance": "Medium"
    }
  ],
  "recommendations": {
    "priority": "Critical",
    "actions": [
      {
        "category": "Engagement",
        "action": "Send a targeted re-engagement email or customized content.",
        "reason": "Customer's usage has dropped by 15.0% over the last 30 days."
      }
    ]
  }
}
```

---

# 🏗️ System Architecture

```text
                         CHURNIQ
                            │
             ┌──────────────┴──────────────┐
             │                             │
             ▼                             ▼
      Next.js Frontend              FastAPI Backend
             │                             │
             │ HTTPS / REST API            │
             └──────────────► /predict ───┤
                                           │
                                    Data Injection
                                     (Logs Fetch)
                                           │
                                           ▼
                                    Preprocessing
                                           │
                                           ▼
                                 Random Forest Model
                                           │
                                           ▼
                                        SHAP
                                           │
                                           ▼
                                   Recommendations
                                           │
                                           ▼
                                      JSON Response
                                           │
             ◄─────────────────────────────┘
             │
             ▼
       Risk + Drivers + Action
```

---

# 🐳 Dockerized Architecture

Both the frontend and backend are containerized.

```text
                    Docker
                       │
          ┌────────────┴────────────┐
          │                         │
          ▼                         ▼
   Frontend Container         Backend Container
       Next.js                    FastAPI
       :3000                       :8000
          │                         │
          └────────── API ──────────┘
                     │
            Random Forest + SHAP
```

### Backend container

Contains:

* FastAPI
* Python dependencies
* preprocessing pipeline
* trained Random Forest model
* SHAP explainer
* recommendation logic

### Frontend container

Contains:

* Next.js
* React
* production build
* frontend dependencies

This allows the ML backend and frontend to be developed, tested, and deployed independently.

---

# ☁️ Deployment

The application is deployed as two services:

```text
Frontend
https://churniq-analysis-next.onrender.com

Backend
https://churniq-fastapi-server.onrender.com
```

---

# 🧰 Tech Stack

| Category         | Technology                                  |
| ---------------- | ------------------------------------------- |
| Programming      | Python, JavaScript                          |
| Machine Learning | Random Forest, XGBoost, Logistic Regression |
| Data Processing  | pandas, NumPy                               |
| ML Pipeline      | scikit-learn, imbalanced-learn              |
| Imbalanced Data  | SMOTE                                       |
| Explainability   | SHAP                                        |
| Backend          | FastAPI                                     |
| API              | REST                                        |
| Frontend         | Next.js, React                              |
| Styling          | CSS                                         |
| Serialization    | joblib                                      |
| Containerization | Docker                                      |
| Deployment       | Render                                      |

---

# 📁 Project Structure

```text
Customer-Churn-analysis/
│
├── churn-backend/
│   ├── data/
│   │   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   │   ├── customer_logs.csv
│   │   └── master_training_data.csv
│   │
│   ├── models/
│   │   ├── churn_model.pkl
│   │   ├── scaler.pkl
│   │   └── feature_columns.pkl
│   │
│   ├── schemas/
│   │   └── prediction.py
│   │
│   ├── services/
│   │   ├── prediction_services.py
│   │   ├── recommendation_service.py
│   │   └── feature_service.py
│   │
│   ├── simulation/
│   │   └── customer_behavior.py
│   │
│   ├── src/
│   │   ├── preprocessing.py
│   │   └── behavioral_features.py
│   │
│   ├── training/
│   │   └── train_churn_model.py
│   │
│   ├── .dockerignore
│   ├── Dockerfile
│   ├── main.py
│   └── requirements.txt
│
├── churn-frontend/
│   ├── src/
│   │   └── app/
│   │       ├── page.jsx
│   │       ├── layout.jsx
│   │       └── globals.css
│   │
│   ├── .dockerignore
│   ├── Dockerfile
│   ├── package.json
│   └── package-lock.json
│
├── notebooks/
├── LICENSE
└── README.md
```

---

# 🚀 Running Locally

## Backend

```bash
cd churn-backend

python -m venv .venv
```

Activate the virtual environment.

### Windows

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run FastAPI:

```bash
uvicorn main:app --reload
```

API:

```text
http://localhost:8000
```

Swagger:

```text
http://localhost:8000/docs
```

---

## Frontend

```bash
cd churn-frontend
npm install
npm run dev
```

Frontend:

```text
http://localhost:3000
```

For local development, configure:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

---

# � Key Engineering Decisions

### Dynamic Behavioral Log Injection
Instead of forcing the frontend to compute telemetry, the FastAPI backend dynamically reads `customerID`, queries the latest behavior logs from the database layer, and joins the arrays immediately before model execution.

### Random Forest over XGBoost
After synthesizing 90-day behavioral telemetry logs with the original demographics, the Random Forest model outperformed XGBoost in both ROC-AUC (0.923 vs 0.911) and precision, leading to its selection as the production model.

### Behavior-Driven Recommendations
Standard churn predictors output generic financial solutions. This pipeline reads the raw SHAP driver arrays, identifies if churn is primarily driven by usage drop-off or excessive support tickets, and dynamically routes the customer to the appropriate mitigation flow (e.g. white-glove support).

### Clean Architecture
The final implementation migrated away from monolithic notebooks into a clean enterprise microservice architecture (`services/`, `training/`, `models/`, `simulation/`), ensuring robust isolation of offline pipelines and live serving layers.

---

# 📊 Original Dataset

**IBM Telco Customer Churn Dataset** combined with **Simulated Behavioral Telemetry**.

Dataset source:

https://www.kaggle.com/datasets/blastchar/telco-customer-churn

---

# 📄 License

This project is licensed under the MIT License.

See [LICENSE](LICENSE) for details.

---

# 👤 Author

**Prabhu-Jethi**

GitHub:
https://github.com/Prabhu-Jethi

LinkedIn:
https://www.linkedin.com/in/prabhudatta-jethi/

---

> **ChurnIQ — Predict the risk. Understand the why. Take action.**
