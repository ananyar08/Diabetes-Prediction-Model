import os
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
from pydantic import BaseModel

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and scaler
scaler = joblib.load("scaler.pkl")
model = joblib.load("rf_model.pkl")

# Input data format
class DiabetesInput(BaseModel):
    Glucose: float
    BMI: float
    Age: int
    BloodPressure: float
    DiabetesPedigreeFunction: float

# Serve index.html (Ensure it is in the same directory as main.py)
@app.get("/")
def serve_ui():
    return FileResponse("index.html")

# Prediction endpoint
@app.post("/predict")
def predict(data: DiabetesInput):
    input_features = np.array([[data.Glucose, data.BMI, data.Age, data.BloodPressure, data.DiabetesPedigreeFunction]])
    input_scaled = scaler.transform(input_features)
    prediction = model.predict(input_scaled)[0]
    return {"diabetes_prediction": int(prediction)}

# Ensure Render binds to the correct host and port
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))  # Use Render's assigned port
    uvicorn.run(app, host="0.0.0.0", port=port)

