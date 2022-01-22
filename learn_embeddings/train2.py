
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data

from models.lstm import FeedForwardNN
from tabular_dataset import TabularDataset

PATH  =  os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


num_classes = 4
embedding_size = 10

embedding = nn.Embedding(num_classes, embedding_size)

class_vector = torch.tensor([1, 0, 3, 3, 2])

embedded_classes = embedding(class_vector)
print(embedded_classes.size()) # => torch.Size([5, 10])

t = 1