import os
import random
import re
import pandas as pd
import numpy as np
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

    def load_img(self, img_path):
        image = np.array(PILImage.open(img_path))
        image_original = self.transform_img(image)
        return image_original

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):

        no_items = len(self.data_df)
        parts = int(no_items / self.seq_length)
        start_sequence = random.randint(1, parts - 1) * self.seq_length

        no_items = len(self.data_df)
        parts = int(no_items / self.seq_length)
        start_sequence = random.randint(1, parts - 1) * self.seq_length

        sequence_ids = [start_sequence + i for i in range(self.seq_length)]

        X_list = []
        y_list = []

        #for i in range(idx, idx + self.seq_length):
        for i in sequence_ids:
            X_list.append(torch.tensor(self.data_df.iloc[i][self.categorical_features].values, dtype=torch.long))
            y_list.append(self.data_df.iloc[i][self.output_feature])

        X_list = torch.stack(X_list, dim=0) 

        return X_list, torch.tensor(y_list)