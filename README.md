# ChurnIQ — Customer Churn Prediction & Retention Intelligence Platform

> An end-to-end, explainable customer churn intelligence platform that predicts customer churn risk, explains the key drivers behind each prediction, and converts model outputs into actionable retention recommendations.

** Live Application --> https://churniq-next.vercel.app
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
Customer Data
      │
      ▼
Preprocessing
      │
      ▼
XGBoost Churn Prediction
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

Telecom companies lose customers for many different reasons, including contract type, tenure, pricing, service configuration, and customer support experience.

The objective of this project is to identify customers with a high probability of churn early enough for a business to take preventive action.

The model is trained using the IBM Telco Customer Churn dataset containing **7,043 customers** and customer/service attributes.

The project prioritizes **recall** because missing an actual churner can be more costly to the business than contacting a customer who ultimately stays.

---

# 💡 Solution

ChurnIQ provides an end-to-end ML workflow:

### 1. Predict

The system accepts customer profile and service information and sends it to a FastAPI backend.

The backend preprocesses the data and generates a churn probability using the trained XGBoost model.

### 2. Explain

SHAP is used to identify the most influential features contributing to an individual prediction.

Instead of returning:

```text
Churn probability: 76%
```

the system can explain:

```text
Month-to-month contract      → increases risk
Short tenure                 → increases risk
Higher monthly charges       → increases risk
```

### 3. Act

The strongest risk signals are converted into retention recommendations so that the prediction can support an actual business response.

---

# 🧠 Machine Learning Pipeline

```text
Raw Customer Data
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
Best Model: XGBoost
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

SMOTE is applied to the training data to address the class imbalance, while scaling is fitted on training data and then applied to the test data. The original project also compares Logistic Regression, Random Forest and XGBoost before selecting XGBoost as the strongest model.

---

# 🏆 Model Performance

| Model               | Precision |   Recall | F1-Score |  ROC-AUC |
| ------------------- | --------: | -------: | -------: | -------: |
| Logistic Regression |      0.62 |     0.79 |     0.69 |     0.84 |
| Random Forest       |      0.65 |     0.81 |     0.72 |     0.88 |
| **XGBoost**         |  **0.68** | **0.85** | **0.75** | **0.90** |

The XGBoost model achieves approximately **0.90 ROC-AUC** with **85% churn recall**, making it the selected production model.

### Why Recall?

The dataset has a significant class imbalance. Optimizing only for accuracy could hide poor churn detection performance.

The project therefore emphasizes:

**Recall → F1 → ROC-AUC**

rather than relying on accuracy alone.

---

# 🔍 Explainable AI with SHAP

ChurnIQ uses **SHAP (SHapley Additive exPlanations)** to explain individual predictions.

The explanation layer allows users to understand:

* Which features increased churn risk
* Which features reduced churn risk
* The relative impact of each feature
* Why a particular customer received their risk score

Important churn signals identified in the project include:

* Contract type
* Tenure
* Monthly charges
* Technical support
* Internet service

The original analysis also identifies contract type and tenure among the strongest churn drivers.

---

# 🖥️ ChurnIQ Frontend

The frontend is built as a modern interactive web application rather than a notebook or static model demonstration.

### Main capabilities

* Customer risk profile
* Dynamic customer inputs
* Churn probability visualization
* Low / Medium / High risk classification
* SHAP driver visualization
* Retention recommendations
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
  "churn_probability": 0.76,
  "churn_percentage": 76.0,
  "risk_level": "high",
  "model": "XGBoost",
  "drivers": [],
  "recommendations": {}
}
```

The exact SHAP driver and recommendation contents depend on the customer profile being evaluated.

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
                                    Preprocessing
                                           │
                                           ▼
                                      XGBoost Model
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
              XGBoost + SHAP
```

### Backend container

Contains:

* FastAPI
* Python dependencies
* preprocessing pipeline
* trained XGBoost model
* SHAP
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

### Production request flow

```text
User
 │
 ▼
ChurnIQ Frontend
 │
 │ POST /predict
 ▼
FastAPI Backend
 │
 ▼
XGBoost
 │
 ▼
SHAP
 │
 ▼
Recommendation Engine
 │
 ▼
Prediction JSON
 │
 ▼
ChurnIQ UI
```

The frontend uses the production API URL through:

```text
NEXT_PUBLIC_API_URL
```

rather than hardcoding a local `localhost` API address.

---

# 🧰 Tech Stack

| Category         | Technology                                  |
| ---------------- | ------------------------------------------- |
| Programming      | Python, JavaScript                          |
| Machine Learning | XGBoost, Random Forest, Logistic Regression |
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
| Version Control  | Git, GitHub                                 |
| Deployment       | Render                                      |

The original repository already contains the underlying Python ML stack, SHAP, Streamlit application, and persisted-model workflow; the project has since been extended into the separate FastAPI + Next.js architecture documented above.

---

# 📁 Project Structure

```text
Customer-Churn-analysis/
│
├── churn-backend/
│   ├── models/
│   │   └── trained model artifacts
│   │
│   ├── schemas/
│   │   └── prediction.py
│   │
│   ├── services/
│   │   ├── prediction_services.py
│   │   └── recommendation_service.py
│   │
│   ├── src/
│   │   └── preprocessing.py
│   │
│   ├── .dockerignore
│   ├── Dockerfile
│   ├── main.py
│   ├── requirements.txt
│   └── test_preprocessing.py
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
├── data/
├── notebooks/
├── apps/
├── src/
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

# 🐳 Running with Docker

## Backend

```bash
cd churn-backend

docker build -t churn-backend .

docker run -d \
  -p 8000:8000 \
  --name churn-backend-container \
  churn-backend
```

## Frontend

```bash
cd churn-frontend

docker build -t churn-frontend .

docker run -d \
  -p 3000:3000 \
  --name churn-frontend-container \
  churn-frontend
```

Then open:

```text
http://localhost:3000
```

---

# 🧪 Testing

The final application can be tested using different customer profiles:

| Test Case                   | Variables                                      |
| --------------------------- | ---------------------------------------------- |
| High-risk profile           | Short tenure + monthly contract                |
| Long-term customer          | High tenure + two-year contract                |
| High-charge customer        | Higher monthly charges                         |
| Contract comparison         | Monthly vs one-year vs two-year                |
| Service comparison          | Different InternetService/service combinations |
| Customer profile comparison | Male/Female + different service configurations |

The important validation is that the frontend sends the changed customer information to the backend and displays the resulting model response, SHAP drivers, and recommendation.

---

# 🔑 Key Engineering Decisions

### SMOTE after train/test split

SMOTE is applied only to the training data to avoid synthetic samples leaking into the test set.

### XGBoost over Random Forest

XGBoost achieved the strongest ROC-AUC and churn recall among the evaluated models.

### SHAP instead of a black-box score

The model prediction is accompanied by feature-level explanations so users can understand the reasoning behind individual predictions.

### FastAPI for model serving

The trained model and preprocessing pipeline are separated from the frontend and exposed through a REST API.

### Next.js for the application layer

The frontend consumes backend responses and provides an interactive interface for business-oriented risk assessment.

### Docker for reproducibility

Both services are containerized so their runtime dependencies are isolated and deployment is consistent across environments.

### Environment-based API configuration

The frontend uses `NEXT_PUBLIC_API_URL` so the same code can communicate with the local FastAPI server during development and the deployed backend in production.

---

# 📌 What This Project Demonstrates

ChurnIQ demonstrates experience across the complete machine-learning application lifecycle:

```text
Data
 ↓
EDA
 ↓
Preprocessing
 ↓
Class Imbalance Handling
 ↓
Model Training
 ↓
Model Evaluation
 ↓
Explainable AI
 ↓
REST API
 ↓
Frontend Application
 ↓
Docker
 ↓
Cloud Deployment
```

This makes the project more than a standalone ML notebook: it demonstrates how a trained model can be transformed into a usable, explainable, deployable application.

---

# 📊 Original Dataset

**IBM Telco Customer Churn Dataset**

* 7,043 customers
* Telecom customer/service attributes
* Binary churn target

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
