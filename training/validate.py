import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, PreTrainedModel

from ..types.declarations import Answers
from .metrics import evaluate_model
from .utils import get_prediction


def validate(
    model: PreTrainedModel,
    tokenizer: BertTokenizer,
    val_loader: DataLoader,
    val_texts: list[str],
    val_questions: list[str],
    val_answers: Answers,
) -> tuple(torch.float64, dict[str, int]):
    """
    Validate our training data findings
    """

    # evaluate model
    model.eval()

    pbar = tqdm(total=len(val_loader))

    total_loss: torch.float64 = 0

    for batch_idx, batch in enumerate(val_loader):
        with torch.no_grad():
            outputs = get_prediction(model, batch)
            loss = outputs[0]
            # find the total loss
            total_loss += loss.item()

        pbar.set_postfix(
            {"Batch": batch_idx + 1, "Loss": round(loss.item(), 3)}, refresh=True
        )

    total_loss /= len(val_loader)

    metrics_dict = evaluate_model(
        model, tokenizer, val_texts, val_questions, val_answers
    )

    model.train()

    return total_loss, metrics_dict
