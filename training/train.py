import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedModel

from .utils import get_prediction


def train_epoch(
    model: PreTrainedModel, train_loader: DataLoader, optim: AdamW, scheduler: LambdaLR
) -> torch.float64:
    """
    Train model epochs
    """
    # Set model to train mode
    model.train()

    total_loss: torch.float64 = 0

    pbar = tqdm(train_loader)

    for batch in pbar:
        optim.zero_grad()

        outputs = get_prediction(model, batch)
        loss = outputs[0]

        # backwards pass
        loss.backward()

        # update weights
        optim.step()

        # update the learning rate
        scheduler.step()

        total_loss += loss.item()

        pbar.set_postfix({"batch loss": loss.item()})

    total_loss /= len(train_loader)

    return total_loss
