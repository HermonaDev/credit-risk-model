from pydantic import BaseModel

class CustomerData(BaseModel):
    recency: float
    frequency: float
    monetary_total: float

class PredictionResponse(BaseModel):
    customer_id: str
    is_high_risk: int
    risk_probability: float
    model_version: str = 'random_forest_v1'
