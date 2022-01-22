import random
import numpy as np
import pandas as pd
import torch
from torch.utils import data
from PIL import Image as PILImage
from torchvision import transforms as transforms

class DatasetTransformer(data.Dataset):

    def __init__(self, **kwargs):
        self.data_path = kwargs['data_path']
        self.seq_length = kwargs['seq_length']
        self.features = kwargs['features']
        self.categorical_features = kwargs['categorical_features']
        self.output_feature = kwargs['output_feature']

        self.features.append(self.output_feature)

        self.data_df = pd.read_csv(self.data_path)

    def __len__(self):
        return int(len(self.data_df) - self.seq_length - 1)

    def __getitem__(self, idx):

        #start = random.randint(0, len(self.data_df) - self.seq_length - 1)
        start = idx
        X = torch.tensor(self.data_df[self.features].loc[start : start + self.seq_length - 1].values)
        X_Emb = torch.tensor(self.data_df[self.categorical_features].loc[start : start + self.seq_length - 1].values, dtype=torch.long)
        #print(start + 1, (start + 1) + self.seq_length -1)
        y = torch.tensor(self.data_df[self.output_feature].loc[start + 1 : (start + 1) + self.seq_length -1].values)

        return X, X_Emb, y