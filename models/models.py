from tokenizers import Encoding
from transformers import (
    AutoTokenizer,
    BertModel,
    ConvBertForQuestionAnswering,
    RobertaForQuestionAnswering,
)

from training.utils import get_device

MODELS = {
    "icebert":  { "name": "Mideind/icebert", "model_type": RobertaForQuestionAnswering},
    "convbert": { "name": "jonfd/convbert-base-igc-is", "model_type": ConvBertForQuestionAnswering}
}

def load(model_name: str) -> tuple[BertModel, Encoding]:
    if model_name not in MODELS: 
        raise ValueError(f"Invalid model name: {model_name}. Supported models: {list(MODELS.keys())}")

    BERT_MODEL = MODELS[model_name]
    BERT_TYPE = BERT_MODEL["model_type"]
    BERT_URL = BERT_MODEL["name"]

    device = get_device()
    model = BERT_TYPE.from_pretrained(BERT_URL).to(device)

    tokenizer = AutoTokenizer.from_pretrained(BERT_URL)

    print(f"\n\t{model_name.capitalize()} is loaded on {device}")

    return model, tokenizer
