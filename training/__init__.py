import torch
import wandb
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from train import train_epoch
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup
from validate import validate

from definitions.declarations import Answers


def run_training(
    train_loader: DataLoader,
    test_loader: DataLoader,
    test_data_raw: tuple[list[str], list[str], list[Answers]],
    model: BertModel,
    tokenizer: BertTokenizer,
    epochs: int,
    lr: float,
) -> None:
    training_steps: int = epochs * len(train_loader)
    warmup_steps: int = training_steps // 10

    optim = AdamW(model.parameters(), lr=lr)

    scheduler: LambdaLR = get_linear_schedule_with_warmup(
        optim, num_warmup_steps=warmup_steps, num_training_steps=training_steps
    )

    test_texts, test_questions, test_answers = test_data_raw
    highest_f1_score = 0

    for epoch in range(epochs):
        print(f"\n****** epoch {epoch + 1}/{epochs} ********\n")
        train_loss = train_epoch(model, train_loader, optim, scheduler)
        val_loss, metrics_dict = validate(
            model, tokenizer, test_loader, test_texts, test_questions, test_answers
        )

        if metrics_dict["f1"] > highest_f1_score:
            highest_f1_score = metrics_dict["f1"]
            PATH = f"path/save/model{epoch}.pt"
            torch.save(model.state_dict(), PATH)

        loss_dict = {"train_loss": train_loss, "val_loss": val_loss}
        wandb.log({**loss_dict, **metrics_dict})
