"""FastAPI deployment for credit risk model."""
import pandas as pd
import mlflow
import mlflow.sklearn
import os
import glob
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

# Function to load the latest MLflow model
def load_latest_model():
    """
    Load the latest trained model from MLflow runs.
    Falls back to a dummy model if no MLflow model found.
    """
    try:
        # Search for model in MLflow runs
        model_patterns = [
            "./mlruns/*/*/artifacts/random_forest_model",
            "mlruns/*/*/artifacts/random_forest_model",
            "./mlruns/*/*/artifacts/logistic_regression_model"
        ]
        
        for pattern in model_patterns:
            model_dirs = glob.glob(pattern)
            if model_dirs:
                latest_model = max(model_dirs, key=os.path.getmtime)
                model = mlflow.sklearn.load_model(latest_model)
                print(f"✅ Model loaded from: {latest_model}")
                return model, os.path.basename(os.path.dirname(latest_model))
    
    except Exception as e:
        print(f"⚠️ MLflow model loading failed: {e}")
    
    # Fallback: create a simple dummy model for API testing
    print("⚠️ Using dummy model for demonstration")
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np
    
    dummy_model = RandomForestClassifier(n_estimators=10, random_state=42)
    # Train on dummy data
    X_dummy = np.random.rand(10, 3)
    y_dummy = np.random.randint(0, 2, 10)
    dummy_model.fit(X_dummy, y_dummy)
    
    return dummy_model, "dummy_model_v1"

# Load model
model, model_version = load_latest_model()

# FastAPI app
app = FastAPI(
    title="Credit Risk Prediction API",
    description="API for predicting credit risk probability using RFM features",
    version="1.0"
)

# Pydantic models (also defined in pydantic_models.py)
class CustomerData(BaseModel):
    recency: float
    frequency: float
    monetary_total: float
    customer_id: Optional[str] = "CUSTOMER_001"

class PredictionResponse(BaseModel):
    customer_id: str
    is_high_risk: int
    risk_probability: float
    model_version: str

# API endpoints
@app.get("/")
def read_root():
    return {
        "message": "Credit Risk Prediction API",
        "status": "active",
        "endpoints": ["/health", "/predict", "/predict_test", "/docs"]
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_version": model_version,
        "api_version": "1.0"
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(data: CustomerData):
    """
    Predict credit risk for a customer.
    
    Parameters:
    - recency: Days since last transaction
    - frequency: Number of transactions
    - monetary_total: Total transaction amount (absolute)
    - customer_id: Optional customer identifier
    
    Returns:
    - Risk prediction and probability
    """
    # Create feature DataFrame
    features = pd.DataFrame([{
        'recency': data.recency,
        'frequency': data.frequency,
        'monetary_total': data.monetary_total
    }])
    
    try:
        # Predict
        probability = model.predict_proba(features)[0, 1]
        prediction = int(probability >= 0.5)
        
        return PredictionResponse(
            customer_id=data.customer_id,
            is_high_risk=prediction,
            risk_probability=round(probability, 4),
            model_version=model_version
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/predict_test")
def predict_test():
    """
    Test endpoint with sample data.
    Useful for quick API verification.
    """
    test_data = {
        "recency": 30.0,
        "frequency": 5.0,
        "monetary_total": 10000.0,
        "customer_id": "TEST_CUSTOMER_001"
    }
    
    # Use the predict endpoint logic
    features = pd.DataFrame([{
        'recency': test_data['recency'],
        'frequency': test_data['frequency'],
        'monetary_total': test_data['monetary_total']
    }])
    
    try:
        probability = model.predict_proba(features)[0, 1]
        return {
            "test_data": test_data,
            "prediction": {
                "risk_probability": round(probability, 4),
                "is_high_risk": int(probability >= 0.5),
                "risk_level": "high" if probability >= 0.5 else "low"
            },
            "model_info": {
                "version": model_version,
                "status": "operational"
            }
        }
    except Exception as e:
        return {
            "test_data": test_data,
            "error": str(e),
            "model_info": {
                "version": model_version,
                "status": "degraded"
            }
        }

@app.get("/model_info")
def model_info():
    """Get information about the loaded model."""
    return {
        "model_type": type(model).__name__,
        "model_version": model_version,
        "features_used": ["recency", "frequency", "monetary_total"],
        "threshold": 0.5
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)