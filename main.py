from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import joblib
import numpy as np
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

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

# Serve the UI
@app.get("/")
def serve_ui():
    return FileResponse("static/index.html")

# Prediction endpoint
@app.post("/predict")
def predict(data: DiabetesInput):
    input_features = np.array([[data.Glucose, data.BMI, data.Age, data.BloodPressure, data.DiabetesPedigreeFunction]])
    input_scaled = scaler.transform(input_features)
    prediction = model.predict(input_scaled)[0]
    return {"diabetes_prediction": int(prediction)}

