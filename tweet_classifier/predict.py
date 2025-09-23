# predict.py
import torch
from tweet_classifier.tokenizer_loader import bert_tokenizer  # tu tokenizer

def predict_tweet(model, tweet):
    device = next(model.parameters()).device

    input_ids, attention_mask = bert_tokenizer(tweet)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask=attention_mask)
        pred = torch.argmax(logits, dim=1).cpu().item()

    return pred