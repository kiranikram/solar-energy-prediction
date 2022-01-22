


import random

import pandas as pd
import torch

data_path = "C:\\Work\\ML\\solar-energy-prediction\\data\\sunrock_clean.csv"

data_df = pd.read_csv(data_path)

start = 2859
seq_length = 20

print(len(data_df))
print(start + 1, (start + 1) + seq_length)

y = torch.tensor(data_df['Total'].loc[start + 1 : (start + 1) + seq_length].values)

print(y.shape)