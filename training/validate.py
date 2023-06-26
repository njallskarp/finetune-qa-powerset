import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, PreTrainedModel

device = torch.device("cpu:0" if torch.cuda.is_available() else "cpu")

def get_prediction(model: PreTrainedModel, batch: DataLoader[tuple[torch.Tensor, torch.Tensor]]) -> PreTrainedModel: 

  input_ids = batch['input_ids'].to(device)
  a_mask = batch['attention_mask'].to(device)
  start_pos = batch['start_positions'].to(device)
  end_pos = batch['end_positions'].to(device)
        
  return model(input_ids, attention_mask=a_mask, start_positions=start_pos, end_positions=end_pos)

def validate(model: PreTrainedModel, tokenizer: BertTokenizer, val_loader: DataLoader, val_texts: list[str], val_questions: list[str], val_answers: list[dict]):

    # evaluate model 
    model.eval()

    pbar = tqdm(total = len(val_loader))

    total_loss: torch.float64 = 0

    for batch_idx, batch in enumerate(val_loader):
        with torch.no_grad(): 
            outputs = get_prediction(model, batch)
            loss = outputs[0]
            # find the total loss
            total_loss += loss.item()

        pbar.set_postfix({'Batch': batch_idx+1, 'Loss': round(loss.item(),3)}, refresh=True)

    total_loss /= len(val_loader)

    #TODO: need to make evaluate_model function
    metrics_dict = evaluate_model(model, tokenizer, val_texts, val_questions, val_answers)

    model.train()

    return total_loss, metrics_dict
