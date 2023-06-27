import torch
from torch.utils.data import Dataset


class SquadDataset(Dataset):
    """
    Code reused from open source colab notebook:
    https://github.com/alexaapo/BERT-based-pretrained-model-using-SQuAD-2.0-dataset/blob/main/Fine_Tuning_Bert.ipynb
    """
    class SquadDataset(Dataset):
        """
        A PyTorch dataset.

        Args:
            encodings (any): The preprocessed input encodings.
            is_train (bool, optional): Whether the dataset is for training or not. Defaults to True.
        """
    def __init__(self, encodings: any, is_train: bool = True):
        self.encodings = encodings
        self.is_train = is_train

    def __getitem__(self, idx: int) -> dict:
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()} 
        return item

    def __len__(self) -> int:
        return len(self.encodings.input_ids)