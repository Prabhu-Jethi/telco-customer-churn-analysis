from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from schemas.prediction import CustomerData
from services.prediction_services import predict_churn
from services.recommendation_service import generate_recommendations

app = FastAPI(
    title="Churn-Analysis API",
    description="Customer churn prediction and retention intelligence API",
    version="1.0.0"
)

@app.get("/")
def root():
    return{
        "message": "App is running",
        "status": "healthy"
    }

@app.api_route("/health", methods=['GET', 'HEAD'])
def health_check():
    return{
        "status": "ok"
    }

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://churniq-analysis-next.onrender.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
def predict(customer: CustomerData):

    customer_dict = customer.model_dump()
    
    prediction, probability, drivers = predict_churn(
        customer_dict
    )
    
    if probability < 0.30:
        risk_level = 'low'
    elif probability < 0.60:
        risk_level = 'mid'
    else:
        risk_level = 'high'

    recommendations = generate_recommendations(
        customer_dict,
        probability,
        drivers
    )

    return{
        "prediction": prediction,
        "churn_probability": round(probability, 4),
        "churn_percentage": round(probability * 100, 2),
        "risk_level": risk_level,
        "model": "Random Forest",
        "drivers": drivers,
        "recommendations": recommendations
    }

