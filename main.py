from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load model
model = joblib.load("legal_classifier.pkl")

# FastAPI app
app = FastAPI()

class TextRequest(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "Legal Classifier is running 🎯"}

@app.post("/predict/")
def predict(request: TextRequest):
    prediction = model.predict([request.text])
    return {"prediction": prediction[0]}
