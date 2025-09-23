# tokenizer_loader.py
from transformers import BertTokenizer
import torch
from .preprocessing import clean_tweet  # importamos limpieza

MAX_LEN = 128  # o el que uses en tu notebook
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

def bert_tokenizer(texts):
    """
    texts: str o list[str]
    Devuelve: input_ids y attention_mask en tensores
    """
    if isinstance(texts, str):
        texts = [texts]

    input_ids = []
    attention_masks = []

    for sent in texts:
        cleaned_sent = clean_tweet(sent)  # primero limpiar el tweet
        encoded_sent = tokenizer.encode_plus(
            text=cleaned_sent,
            add_special_tokens=True,
            max_length=MAX_LEN,
            padding='max_length',
            truncation=True,
            return_attention_mask=True
        )
        input_ids.append(encoded_sent['input_ids'])
        attention_masks.append(encoded_sent['attention_mask'])

    # Convertir a tensores
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks