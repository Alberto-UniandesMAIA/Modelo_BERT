# model_loader.py
import torch
import torch.nn as nn
from transformers import BertModel
import warnings

# Detectar dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Definir la clase del modelo ---
class Bert_Classifier(nn.Module):
    def __init__(self, freeze_bert=False):
        super(Bert_Classifier, self).__init__()
        n_input = 768
        n_hidden = 50
        n_output = 5  # número de clases

        # Cargar BERT preentrenado
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # Clasificador
        self.classifier = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_output)
        )

        # Congelar BERT si es necesario
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state_cls = outputs[0][:, 0, :]
        logits = self.classifier(last_hidden_state_cls)
        return logits

# --- Función para cargar modelo desde .pth ---
def load_model(model_path):
    model = Bert_Classifier()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # silencia warnings de Transformers
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model