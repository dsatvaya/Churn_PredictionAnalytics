import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from transformer import ChurnFeatureEngineer

# Initialize API
app = FastAPI(title="Customer Churn Prediction API", description="Serves batch predictions for downstream reporting teams")

# Define the absolute path and load the model dynamically
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'churn_model_v1.pkl')

try:
    model_pipeline = joblib.load(model_path)
    print(f"Successfully loaded model from {model_path}")
except Exception as e:
    print(f"Warning: Model not loaded. Ensure pipeline_build.py has been executed. Error: {e}")

# Define the expected input schema strictly matching your raw dataset headers
class CustomerRecord(BaseModel):
    customerID: str
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: str  
    monthly: Optional[str] = None  

@app.get("/")
def health_check():
    return {
        "API_Status": "Active", 
        "Model_Loaded": "churn_model_v1.pkl",
        "Documentation": "Navigate to /docs to interact with the endpoints."
    }

@app.post("/predict_batch")
def predict_batch(records: list[CustomerRecord]):
    try:
        raw_data = [record.dict() for record in records]
        input_df = pd.DataFrame(raw_data)
        
        customer_ids = input_df['customerID'].tolist()
        clean_df = input_df.drop(['customerID', 'monthly'], axis=1, errors='ignore')
        
        probabilities = model_pipeline.predict_proba(clean_df)[:, 1]
        
        BUSINESS_THRESHOLD = 0.52
        
        results = []
        for i in range(len(probabilities)):
            prob = float(probabilities[i])
            custom_prediction = 1 if prob >= BUSINESS_THRESHOLD else 0
            
            results.append({
                "customerID": customer_ids[i],
                "churn_prediction": custom_prediction,
                "churn_probability": round(prob, 4)
            })
            
        # THIS IS THE LINE YOU DELETED. IT IS MANDATORY.
        return {"status": "success", "batch_size": len(results), "predictions": results}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")