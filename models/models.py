from tokenizers import Encoding
from transformers import (
    AutoTokenizer,
    BertModel,
    ConvBertForQuestionAnswering,
    RobertaForQuestionAnswering,
)

from training.utils import get_device

MODELS = {
    "icebert": {"name": "Mideind/icebert", "bert_type": RobertaForQuestionAnswering},
    "convbert": {
        "name": "jonfd/convbert-base-igc-is",
        "bert_type": ConvBertForQuestionAnswering,
    },
}


def load(model_name: str) -> tuple[BertModel, Encoding]:
    if model_name not in MODELS:
        raise ValueError(
            f"Invalid model name: {model_name}. Supported models: {list(MODELS.keys())}"
        )

    bert_model = MODELS[model_name]
    bert_type = bert_model["bert_type"]
    bert_url = bert_model["name"]

    device = get_device()
    model = bert_type.from_pretrained(bert_url).to(device)

    tokenizer = AutoTokenizer.from_pretrained(bert_url)

    print(f"\n\t{model_name.capitalize()} is loaded on {device}")

    return model, tokenizer
