import argparse
import csv
import datetime
import logging
import os
#from joblib import load
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import  DataLoader
from datasets.dataset_transformer import DatasetTransformer
from lib.constants import *
from models.transformer import Transformer
from sklearn.metrics import mean_squared_error
from math import sqrt

PATH  =  os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
LOG_LEVEL = logging.getLevelName('DEBUG')
logging.basicConfig(level=LOG_LEVEL)

def get_args_parser():
    parser = argparse.ArgumentParser('Solar panel production prediction', add_help=False)
    parser.add_argument('--env', type=str, default="laptop", help='Enviroment [default: laptop]')
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--seq_length', default=1, type=int)

    return parser.parse_args()
    
def main():

    args = get_args_parser()

    APPLICATION = 'transfomer_{}_{}'.format(args.epochs, MODEL_NUM_LAYERS)
    data_path = os.path.join(PATH, 'data', 'sunrock_clean_spring_all.csv')

    model = Transformer(feature_size=MODEL_FEATURE_SIZE, num_layers=MODEL_NUM_LAYERS, dropout=0, emb_sizes=EMBEDDING_SIZES_TRANSFORMERS).double().to(DEVICE)

    model_path = os.path.join('saved_models', "{}.pth".format(APPLICATION))
    model.load_state_dict(torch.load(model_path))
    model = model.to(DEVICE)
    model.eval()

    df = pd.read_csv(data_path)
    df['DateTime'] = pd.to_datetime(df['DateTime'])

    #dt_march = df[(df['DateTime'] >= datetime.datetime(2018, 3, 1)) & (df['DateTime'] <= datetime.datetime(2018, 3, 31))]

    dt_march = df[(df['DateTime'].dt.month == 4)]
    dt_march['Total'] = dt_march['Total'].apply(lambda x: x * 100)
    march_totals = dt_march['Total'].tolist()


    dt_april = df[(df['DateTime'].dt.month == 4)]
    dt_april.iloc[0, df.columns.get_loc('Total')] = 0


    index = 0
    predictions = []
    y_1 = torch.tensor([0]).double().to(DEVICE)

    for idx_ptu, row_ptu in dt_april.iterrows():

        if(index > 0):
            y_1 = torch.tensor([row_ptu.Total]).double().to(DEVICE)

        X = torch.tensor([row_ptu.sin_day, row_ptu.cos_day, row_ptu.sin_month, row_ptu.cos_month, row_ptu.sin_hour, row_ptu.cos_hour, row_ptu.sin_min, row_ptu.cos_min, y_1])
        X = X.float().to(DEVICE)
        X_Emb = torch.tensor([row_ptu.day, row_ptu.month, row_ptu.hour, row_ptu.minute]).long().to(DEVICE)

        X= X.unsqueeze(0).unsqueeze(0)
        X_Emb = X_Emb.unsqueeze(0).unsqueeze(0)
        
        y_1 = model(X, X_Emb, DEVICE)

        predictions.append(y_1.item()  * 100)

        if(idx_ptu == 0 and index < len(df) - 1):
            df.iloc[index + 1, df.columns.get_loc('Total')] = y_1.item()

        index += 1

    x_labels = dt_april['DateTime'].tolist()
    x_labels_spaced = []
    i = 0
    for label in x_labels:
        if i%96==0:
            x_labels_spaced.append(label)
        else:
            x_labels_spaced.append('')
        i += 1

    #Normalise the data

    plt.title('April 2020 vs. Forecasted April 2021')
    plt.autoscale(axis='x', tight=True)
    plt.xticks(ticks=range(0,len(x_labels_spaced)) ,labels=x_labels_spaced, rotation = 45)
    plt.plot(march_totals, label='April 2020')
    plt.plot(predictions, label='Forecasted April 2021')
    plt.legend()
    plt.show()

    x =1

  
if __name__ == "__main__":
    main()

