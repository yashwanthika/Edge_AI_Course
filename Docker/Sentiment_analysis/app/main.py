from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

nlp = pipeline("sentiment-analysis")
app = FastAPI()

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict(input: TextInput):
    result = nlp(input.text)[0]
    return {"label": result["label"], "score": result["score"]}
