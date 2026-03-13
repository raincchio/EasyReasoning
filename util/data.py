import pandas as pd
from torch.utils.data import Dataset
import torch

class SudokuDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.questions = torch.tensor(
            df['question'].apply(lambda s: [int(c) for c in s]).tolist(),
            dtype=torch.long
        )
        self.answers = torch.tensor(
            df['answer'].apply(lambda s: [int(c) for c in s]).tolist(),
            dtype=torch.long
        )

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        return self.questions[idx], self.answers[idx]

class SudokuminiDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.questions = torch.tensor(
            df['question'].apply(lambda s: [int(c)+1 for c in s]).tolist(),
            dtype=torch.long
        )
        self.answers = torch.tensor(
            df['answer'].apply(lambda s: [int(c) for c in s]).tolist(),
            dtype=torch.long
        )

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        return self.questions[idx], self.answers[idx]


if __name__=='__main__':
    data = SudokuDataset(
        '../data/train_convert.csv')
    print(data)