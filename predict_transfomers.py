import argparse
import logging
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import  DataLoader
from sklearn.preprocessing import StandardScaler

from datasets.dataset_solar import DatasetSolar
from datasets.dataset_transformer import DatasetTransformer
from models.lstm import LSTM
from lib import plot_utils
from models.transformer import Transformer

PATH  =  os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
LOG_LEVEL = logging.getLevelName('DEBUG')
logging.basicConfig(level=LOG_LEVEL)

def get_args_parser():
    parser = argparse.ArgumentParser('Solar panel production prediction', add_help=False)
    parser.add_argument('--env', type=str, default="laptop", help='Enviroment [default: laptop]')
    parser.add_argument('--lr', default=0.0001 , type=float)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--seq_length', default=1, type=int)
    parser.add_argument('--num_layers', default=4, type=int)

    return parser.parse_args()
    
def main():

    args = get_args_parser()

    APPLICATION = 'transfomer_{}_{}'.format(args.epochs, args.num_layers)
    data_path = os.path.join(PATH, 'data', 'sunrock_clean.csv')
    features = ['sin_day', 'cos_day', 'sin_month', 'cos_month', 'sin_hour', 'cos_hour', 'sin_min', 'cos_min']
    categorical_features = ["day", "month", "hour", "minute"]
    output_feature = "Total"

    # Changed embedding (31, 16) = > (31, 15)
    emb_sizes = [(31, 15), (12, 6), (24, 12), (4, 2)]
    learning_rate = args.lr 
    weight_decay = 1e-4
    epochs = args.epochs

    seq_length = args.seq_length

    # parameters for the dataset
    dataset_params = {
        'data_path': data_path,
        'seq_length' : seq_length,
        'device' : DEVICE,
        'features' : features,
        'categorical_features' : categorical_features,
        'output_feature' : output_feature
    }
    
    train_dataset = DatasetTransformer(**dataset_params)
    dataloader_params = {'shuffle': False, 'batch_size': args.batch_size}
    test_loader = DataLoader(train_dataset, **dataloader_params)

    model = Transformer(feature_size=44, num_layers=args.num_layers, dropout=0, emb_sizes=emb_sizes).double().to(DEVICE)

    epochs = args.epochs
    model_path = os.path.join('saved_models', "{}_{}.pth".format(APPLICATION, epochs))
    model.load_state_dict(torch.load(model_path))
    model = model.to(DEVICE)
    model.eval()

    future_preditions = []
    actual_results = []

    for batch in test_loader:
        X, X_Emb, y = batch
        with torch.no_grad():

            X, X_Emb, y = batch
            X = X.to(DEVICE).float()
            X_Emb = X_Emb.to(DEVICE)
            y = y.to(DEVICE).double()

            output = model(X, X_Emb, DEVICE) 

            output= output.view(-1).item()
            y = y.view(-1).item()

            #print(y_t1* 100, ':', output* 100)
            actual_results.append(y * 100)
            future_preditions.append(output * 100) 

    plt.title('Power production prediction')
    plt.ylabel('Solar energy production (kWh).')
    plt.grid(True)
    plt.autoscale(axis='x', tight=True)

    plt.plot(actual_results, label='Ground truth')
    plt.plot(future_preditions, label='Predictions')
    plt.legend()
    plt.show()

    p = 1

if __name__ == "__main__":
    main()

