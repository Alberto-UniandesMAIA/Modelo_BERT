# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from tweet_classifier.model_loader import load_model
from tweet_classifier.predict import predict_tweet  # nombre correcto de la f>
app = FastAPI()
model = load_model("model_weights.pth")

class Tweet(BaseModel):
    text: str

@app.post("/predict")
def predict(tweet: Tweet):
    prediction = predict_tweet(model, tweet.text)
    return {"prediction": prediction}