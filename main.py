
import joblib
import pandas as pd

from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Project directory
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "crop_model.pkl"

# Load the trained model
saved_model = joblib.load(MODEL_PATH)

model = saved_model["model"]
features = saved_model["features"]

# Create FastAPI application
app = FastAPI(
    title="Crop Recommendation API",
    description="Predict a suitable crop using soil and weather data.",
    version="1.0.0",
)


# Input data validation
class CropInput(BaseModel):
    N: float = Field(ge=0, le=200, description="Nitrogen content")
    P: float = Field(ge=0, le=200, description="Phosphorus content")
    K: float = Field(ge=0, le=300, description="Potassium content")
    temperature: float = Field(ge=-10, le=60, description="Temperature in Celsius")
    humidity: float = Field(ge=0, le=100, description="Relative humidity")
    ph: float = Field(ge=0, le=14, description="Soil pH")
    rainfall: float = Field(ge=0, le=1000, description="Rainfall in mm")


# Home endpoint
@app.get("/")
def home():
    return {
        "message": "Crop Recommendation API is running!",
        "docs": "/docs",
        "prediction_endpoint": "/predict",
    }


# Health check endpoint
@app.get("/health")
def health():
    return {"status": "healthy"}


# Crop prediction endpoint
@app.post("/predict")
def predict_crop(data: CropInput):
    try:
        # Convert input into a dictionary
        input_data = data.model_dump()

        # Create a DataFrame with the correct feature names and order
        input_values = pd.DataFrame(
            [[input_data[name] for name in features]],
            columns=features,
        )

        # Predict the crop
        prediction = model.predict(input_values)[0]

        return {
            "predicted_crop": str(prediction),
            "input": input_data,
        }

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="An error occurred while predicting the crop.",
        )
