import numpy as np
import pandas as pd
import torch
from torch.utils import data
from PIL import Image as PILImage
from torchvision import transforms as transforms

class DatasetSolar(data.Dataset):

    def __init__(self, **kwargs):
        self.data_path = kwargs['data_path']
        self.seq_length = kwargs['seq_length']
        self.categorical_features = kwargs['categorical_features']
        self.output_feature = kwargs['output_feature']

        self.data_df = pd.read_csv(self.data_path)
        self.sliding_windows_X = np.array(list(self.window(self.data_df[self.categorical_features].values, self.seq_length)))
        self.sliding_windows_Y = np.array(list(self.window(self.data_df[self.output_feature].values, self.seq_length)))

    def window(self, iterable, size=2):
        i = iter(iterable)
        win = []
        for e in range(0, size):
            win.append(next(i))
        yield win
        for e in i:
            win = win[1:] + [e]
            yield win

    def __len__(self):
        return len(self.sliding_windows_X) - 1

    def __getitem__(self, idx):

        X = torch.tensor(self.sliding_windows_X[idx], dtype=torch.long)
        y = torch.tensor(self.sliding_windows_Y[idx])
        y_t1 = torch.tensor(self.sliding_windows_Y[idx + 1][-1])

        return X, y, y_t1