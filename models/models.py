from transformers import (
    ConvBertForQuestionAnswering,
    RobertaForQuestionAnswering,
)

MODELS = {
    "icebert": {"name": "Mideind/icebert", "bert_type": RobertaForQuestionAnswering},
    "convbert": {
        "name": "jonfd/convbert-base-igc-is",
        "bert_type": ConvBertForQuestionAnswering,
    },
}
