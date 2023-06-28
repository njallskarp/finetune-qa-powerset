import torch
from torch import device
from torch.utils.data import DataLoader
from transformers import BertForQuestionAnswering


def get_device() -> device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_prediction(model: BertForQuestionAnswering, batch: DataLoader[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.FloatTensor]: 
  """
  Get the answer prediction from the model
  """
  device = get_device()

  input_ids = batch['input_ids'].to(device)
  a_mask = batch['attention_mask'].to(device)
  start_pos = batch['start_positions'].to(device)
  end_pos = batch['end_positions'].to(device)
        
  return model(input_ids, attention_mask=a_mask, start_positions=start_pos, end_positions=end_pos)
