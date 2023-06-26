from torch.utils.data import DataLoader, Dataset
import torch
import random

class SquadDataset(Dataset):

    def __init__(self, encodings, is_train = True):
        self.encodings = encodings
        self.is_train = is_train
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # if self.is_train:
        #     item['start_positions'] = torch.tensor(self.start_positions[idx])
        #     item['end_positions'] = torch.tensor(self.end_positions[idx])
        return item
    
    def __len__(self):
        return len(self.encodings.input_ids)

def get_data(dataset_names, model, tokenizer, batch_size):
    # TODO: add more datasets
    pass
