from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from fastapi.middleware.cors import CORSMiddleware
from custom_transformers import DenseTransformer 

# Load model
model = joblib.load("legal_classifier_smote.pkl")

# FastAPI app
app = FastAPI()

# CORS middleware setup to allow all origins and methods
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

class TextRequest(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "Legal Classifier is running 🎯"}

@app.post("/predict/")
def predict(request: TextRequest):
    prediction = model.predict([request.text])
    return {"prediction": prediction[0]}
