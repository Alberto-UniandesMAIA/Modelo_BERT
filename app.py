# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from tweet_classifier.model_loader import load_model
from tweet_classifier.predict import predict_tweet

app = FastAPI()

# Cargar el modelo
model = load_model("model_weights.pth")

# Diccionario de mapeo: número → etiqueta
label_map = {
    0: "religion",
    1: "age",
    2: "ethnicity",
    3: "gender",
    4: "not bullying"
}

class Tweet(BaseModel):
    text: str

@app.post("/predict")
def predict(tweet: Tweet):
    prediction = predict_tweet(model, tweet.text)   # esto devuelve el número
    label = label_map.get(prediction, "unknown")    # convierte a texto
    return {
        "prediction": int(prediction),  # opcional: el número de clase
        "label": label                  # el nombre de la clase
    }
