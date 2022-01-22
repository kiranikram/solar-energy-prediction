import argparse
import logging
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
from torch.utils.data import  DataLoader
from sklearn.preprocessing import StandardScaler

from datasets.dataset_solar import DatasetSolar
from models.lstm import LSTM
from lib import plot_utils, utils

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
    data_path = os.path.join(PATH, 'data', 'sunrock_clean_test.csv')
    data_path_dates = os.path.join(PATH, 'data', 'sunrock_clean.csv')
    data_path_raw = os.path.join(PATH, 'data', 'sunrock_raw_april.csv')

    df = pd.read_csv(data_path)
    data = df.values
    df_dates_to_predict = pd.read_csv(data_path_dates)
    dates_to_predict = df_dates_to_predict.values

    df_raw = pd.read_csv(data_path_raw)

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

    future_preditions = data[:, 4].tolist()
    actual_results = dates_to_predict[:, 4]


    for i in range(len(dates_to_predict) - seq_length - 1):

        X = data [:, 0:4] 
        y = data [:, 4]
        with torch.no_grad():
            X = torch.Tensor(X).to(DEVICE).long()
            y = torch.Tensor(np.array(y)).to(DEVICE).float()

            X = X.unsqueeze(0)
            y = y.unsqueeze(0)

            next_prediction = model(X, y) 
            next_reading_dt = dates_to_predict[i + seq_length + 1, :]

            next_prediction = next_prediction.item()

            # append output to data
            next_data = np.array([[next_reading_dt[0], next_reading_dt[1], next_reading_dt[2], next_reading_dt[3], next_prediction]])

            data = np.append(data, next_data, axis=0)

            data = np.delete(data, 0, axis=0)

            future_preditions.append(next_prediction) 


    actual_results = actual_results * 100
    future_preditions = [x * 100 for x in future_preditions]

    last_prediction = future_preditions[-1]
    future_preditions.append(last_prediction)




    predictions_path = os.path.join(PATH, 'data', 'predictions.csv')
    with open(predictions_path, 'w', encoding='UTF8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['DateTime', 'ptu'])
        for index, row in df_raw.iterrows():
            writer.writerow([row[0], future_preditions[index]])


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

