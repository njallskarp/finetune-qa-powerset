import torch
from torch import device
from torch.utils.data import DataLoader
from transformers import PreTrainedModel


def get_rediction() -> device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_prediction(model: PreTrainedModel, batch: DataLoader[tuple[torch.Tensor, torch.Tensor]]) -> torch.FloatTensor: 
  """
  Get the answer prediction from the model
  """
  device = get_rediction()

  input_ids = batch['input_ids'].to(device)
  a_mask = batch['attention_mask'].to(device)
  start_pos = batch['start_positions'].to(device)
  end_pos = batch['end_positions'].to(device)
        
  return model(input_ids, attention_mask=a_mask, start_positions=start_pos, end_positions=end_pos)
