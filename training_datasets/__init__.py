import torch


class SquadDataset(torch.utils.data.Dataset):
    """
    Code reused from open source colab notebook:
    https://github.com/alexaapo/BERT-based-pretrained-model-using-SQuAD-2.0-dataset/blob/main/Fine_Tuning_Bert.ipynb
    """
    def __init__(self, encodings, is_train = True):
        self.encodings = encodings
        self.is_train = is_train

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()} 
        return item

    def __len__(self):
        return len(self.encodings.input_ids)