import warnings
from tweet_classifier.model_loader import load_model, device
from tweet_classifier.predict import predict_tweet
from tweet_classifier.tokenizer_loader import bert_tokenizer  # o tokenizer.py si ese es tu archivo

# Silenciar warning de flash attention de Transformers
warnings.filterwarnings("ignore", message=".*flash attention.*")

# Cargar modelo desde el .pth
model = load_model("model_weights.pth")

# Test de predicción
tweet = "This is a test tweet!"
pred = predict_tweet(model, tweet)
print(tweet)
print("Predicción:", pred)
