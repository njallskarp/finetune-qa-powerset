import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedModel

device = torch.device("cpu:0" if torch.cuda.is_available() else "cpu")

def set_model(model: PreTrainedModel, batch:DataLoader[tuple[torch.Tensor, torch.Tensor]]) -> PreTrainedModel: 
  input_ids = batch['input_ids'].to(device)
  a_mask = batch['attention_mask'].to(device)
  start_pos = batch['start_positions'].to(device)
  end_pos = batch['end_positions'].to(device)
        
  return model(input_ids, attention_mask=a_mask, start_positions=start_pos, end_positions=end_pos)

    

def train_epoch(model: PreTrainedModel, train_loader: DataLoader, optim: AdamW, scheduler: LambdaLR) -> torch.float64:
    
    # Set model to train mode
    model.train()

    total_loss: torch.float64 = 0
    
    pbar = tqdm(train_loader)

    for batch in pbar: 

        optim.zero_grad()

        outputs = set_model(model, batch)
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
