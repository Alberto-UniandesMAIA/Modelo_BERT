import nltk

nltk.data.path.append('/home/ubuntu/nltk_data')  # Asegura que NLTK busque en la carpeta personalizada
nltk.download('stopwords', download_dir='/home/ubuntu/nltk_data', quiet=True)
nltk.download('punkt_tab', download_dir='/home/ubuntu/nltk_data', quiet=True)
nltk.download('wordnet', download_dir='/home/ubuntu/nltk_data', quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

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
