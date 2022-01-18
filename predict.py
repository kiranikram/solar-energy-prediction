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

from dataset_solar import DatasetSolar
from model import LSTM
from lib import plot_utils

PATH  =  os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
LOG_LEVEL = logging.getLevelName('DEBUG')
logging.basicConfig(level=LOG_LEVEL)

def get_args_parser():
    parser = argparse.ArgumentParser('Solar panel production prediction', add_help=False)
    parser.add_argument('--env', type=str, default="laptop", help='Enviroment [default: laptop]')
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--seq_length', default=24, type=int)
    parser.add_argument('--lstm_hidden_size', default=1024, type=int)
    parser.add_argument('--lstm_num_layers', default=5, type=int)

    return parser.parse_args()
    
def main():

    args = get_args_parser()

    APPLICATION = 'lstm_{}_{}'.format(args.lstm_hidden_size, args.lstm_num_layers)
    data_path = os.path.join(PATH, 'data', 'sunrock_clean.csv')
    emb_sizes = [(31, 16), (12, 6), (24, 12), (4, 2)]
    categorical_features = ["day", "month", "hour", "minute"]
    output_feature = "Total"

    seq_length = args.seq_length

    # parameters for the dataset
    dataset_params = {
        'data_path': data_path,
        'seq_length' : seq_length,
        'device' : DEVICE,
        'categorical_features' : categorical_features,
        'output_feature' : output_feature
    }

    
    train_dataset = DatasetSolar(**dataset_params)
    dataloader_params = {'shuffle': False, 'batch_size': args.batch_size}
    test_loader = DataLoader(train_dataset, **dataloader_params)

    model = LSTM(
        lstm_input_size=37,
        lstm_hidden_size=args.lstm_hidden_size,
        lstm_num_layers=args.lstm_num_layers,
        emb_sizes=emb_sizes,
        num_outputs=1,
        dropout_prob=0.1,
        device=DEVICE
    )

    epochs = args.epochs
    model_path = os.path.join('saved_models', "{}_{}.pth".format(APPLICATION, epochs))
    model.load_state_dict(torch.load(model_path))
    model = model.to(DEVICE)
    model.eval()

    future_preditions = []
    actual_results = []

    for batch in test_loader:
        X, y, y_t1 = batch
        with torch.no_grad():
            X = X.to(DEVICE)
            y = y.to(DEVICE)
            y = y.float()
            y_t1 = y_t1.to(DEVICE).squeeze(0)
            y_t1 = y_t1.float()

            output = model(X, y) 

            #print(y_t1* 100, ':', output* 100)
            actual_results.append(y_t1 * 100)
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

