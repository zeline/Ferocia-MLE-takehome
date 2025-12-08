"""
FastAPI application for term deposit prediction.

This API provides a /predict endpoint that takes raw customer data
and returns a prediction on whether they will subscribe to a term deposit.
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional
import joblib
import numpy as np
from pathlib import Path
import sys


# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from data.preprocessing import preprocess_for_inference

# Initialize FastAPI app
app = FastAPI(
    title="Term Deposit Prediction API",
    description="Predicts whether a customer will subscribe to a term deposit based on their profile and campaign data.",
    version="1.0.0"
)

# Load model and preprocessor at startup
ARTIFACTS_DIR = Path(__file__).parent.parent.parent / "artifacts"

model = None
preprocessor = None

def load_artifacts():
    """Load model and preprocessor from disk."""
    global model, preprocessor
    
    model_path = ARTIFACTS_DIR / "model.pkl"
    preprocessor_path = ARTIFACTS_DIR / "preprocessor.pkl"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Run train.py first.")
    if not preprocessor_path.exists():
        raise FileNotFoundError(f"Preprocessor not found at {preprocessor_path}. Run train.py first.")
    
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    print(f"✅ Loaded model and preprocessor from {ARTIFACTS_DIR}")

# Request schema - matches the raw input format
class CustomerData(BaseModel):
    """Schema for customer input data matching the dataset features."""
    
    # Numeric features
    age: int = Field(..., ge=18, le=100, description="Customer age")
    balance: int = Field(..., description="Average yearly balance in euros")
    day: int = Field(..., ge=1, le=31, description="Last contact day of the month")
    duration: int = Field(..., ge=0, description="Last contact duration in seconds")
    campaign: int = Field(..., ge=1, description="Number of contacts during this campaign")
    pdays: int = Field(..., ge=-1, description="Days since last contact from previous campaign (-1 = not contacted)")
    previous: int = Field(..., ge=0, description="Number of contacts before this campaign")
    
    # Categorical features
    job: str = Field(..., description="Type of job (e.g., admin., blue-collar, technician)")
    marital: str = Field(..., description="Marital status (married, divorced, single)")
    education: str = Field(..., description="Education level (primary, secondary, tertiary, unknown)")
    contact: str = Field(..., description="Contact communication type (cellular, telephone, unknown)")
    month: str = Field(..., description="Last contact month (jan, feb, mar, etc.)")
    poutcome: str = Field(..., description="Outcome of previous campaign (success, failure, other, unknown)")
    
    # Binary features
    default: str = Field(..., description="Has credit in default? (yes/no)")
    housing: str = Field(..., description="Has housing loan? (yes/no)")
    loan: str = Field(..., description="Has personal loan? (yes/no)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "age": 35,
                "job": "management",
                "marital": "married",
                "education": "tertiary",
                "default": "no",
                "balance": 1500,
                "housing": "yes",
                "loan": "no",
                "contact": "cellular",
                "day": 15,
                "month": "may",
                "duration": 180,
                "campaign": 2,
                "pdays": -1,
                "previous": 0,
                "poutcome": "unknown"
            }
        }
#response schema
class PredictionResponse(BaseModel):
    """Schema for prediction response."""
    
    prediction: str = Field(..., description="Predicted outcome: 'yes' or 'no'")
    probability: float = Field(..., description="Probability of subscribing (0-1)")
    confidence: str = Field(..., description="Confidence level: 'low', 'medium', or 'high'")

#loading artifacts on startup - saves time n resources
@app.on_event("startup")
async def startup_event():
    """Load artifacts when the API starts."""
    load_artifacts()

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main UI."""
    html_path = Path(__file__).parent / "index.html"
    return html_path.read_text()


@app.get("/api")
async def api_info():
    """API information endpoint."""
    return {
        "message": "Term Deposit Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Make a prediction",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None
    }
#the predict endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(customer: CustomerData):
    """
    Predict whether a customer will subscribe to a term deposit.
    
    Takes raw customer data and returns a prediction with probability.
    """
    if model is None or preprocessor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs."
        )
    
    try:
        # Convert request to dict
        input_data = customer.model_dump()
        
        # Preprocess the input (same as training)
        X = preprocess_for_inference(input_data, preprocessor)
        
        # Make prediction
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0][1]  # Probability of 'yes'
        
        # Determine confidence level
        if probability < 0.3 or probability > 0.7:
            confidence = "high"
        elif probability < 0.4 or probability > 0.6:
            confidence = "medium"
        else:
            confidence = "low"
        
        return PredictionResponse(
            prediction="yes" if prediction == 1 else "no",
            probability=round(float(probability), 4),
            confidence=confidence
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

# For running directly with: python app.py
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

