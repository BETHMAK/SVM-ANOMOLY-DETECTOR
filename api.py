from fastapi import FastAPI
import pandas as pd
import joblib
from pydantic import BaseModel

# ✅ Load the trained SVM model
MODEL_PATH = r"C:\Users\centi\Desktop\DATASET\SVM_Anomaly_Detection\svm_model.pkl"
svm_model = joblib.load(MODEL_PATH)

# ✅ FastAPI app
app = FastAPI()

# ✅ Define input schema
class InputData(BaseModel):
    features: list[float]  # A list of numerical feature values

# ✅ Prediction endpoint
@app.post("/predict")
def predict(data: InputData):
    # Convert input to DataFrame
    df = pd.DataFrame([data.features])

    # Make prediction
    prediction = svm_model.predict(df)[0]  # Returns 0 (normal) or 1 (attack)

    return {"prediction": int(prediction), "label": "Attack" if prediction == 1 else "Normal"}

# ✅ Root endpoint
@app.get("/")
def root():
    return {"message": "Anomaly Detection API is running!"}
