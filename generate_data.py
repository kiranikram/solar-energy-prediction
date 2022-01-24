import argparse
from calendar import month
import csv
import logging
import os
from joblib import load
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import  DataLoader
from datasets.dataset_transformer import DatasetTransformer
from lib.constants import *
from models.transformer import Transformer

PATH  =  os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
LOG_LEVEL = logging.getLevelName('DEBUG')
logging.basicConfig(level=LOG_LEVEL)

def get_args_parser():
    parser = argparse.ArgumentParser('Solar panel production prediction', add_help=False)
    parser.add_argument('--env', type=str, default="laptop", help='Enviroment [default: laptop]')
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--seq_length', default=1, type=int)
    parser.add_argument('--num_layers', default=4, type=int)

    return parser.parse_args()
    
def main():

    args = get_args_parser()

    APPLICATION = 'transfomer_{}_{}'.format(args.epochs, args.num_layers)
    data_path = os.path.join(PATH, 'data', 'sunrock_clean.csv')

    # parameters for the dataset
    dataset_params = {
        'data_path': data_path,
        'seq_length' : args.seq_length,
        'device' : DEVICE,
        'features' : TRANSFOMERS_FEATURES,
        'categorical_features' : CATEGORICAL_FEATURES,
        'output_feature' : OUTPUT_FEATURE
    }
    
    train_dataset = DatasetTransformer(**dataset_params)
    dataloader_params = {'shuffle': False, 'batch_size': args.batch_size}
    test_loader = DataLoader(train_dataset, **dataloader_params)

    model = Transformer(feature_size=44, num_layers=args.num_layers, dropout=0, emb_sizes=EMBEDDING_SIZES_TRANSFORMERS).double().to(DEVICE)

    model_path = os.path.join('saved_models', "{}.pth".format(APPLICATION))
    model.load_state_dict(torch.load(model_path))
    model = model.to(DEVICE)
    model.eval()

    future_preditions = []
    actual_results = []

    for batch in test_loader:
        with torch.no_grad():

            X, X_Emb, y = batch
            X = X.to(DEVICE).float()
            X_Emb = X_Emb.to(DEVICE)
            y = y.to(DEVICE).double()

            preditions = model(X, X_Emb, DEVICE) 

            preditions= preditions.view(-1).item()
            y = y.view(-1).item()

            actual_results.append(y * 100)
            future_preditions.append(preditions * 100) 

    label_encoder_day = load(os.path.join(PATH, 'encoders', 'label_encoder_day.joblib'))
    label_encoder_hour = load(os.path.join(PATH, 'encoders', 'label_encoder_hour.joblib'))
    label_encoder_minute = load(os.path.join(PATH, 'encoders', 'label_encoder_minute.joblib'))
    label_encoder_month = load(os.path.join(PATH, 'encoders', 'label_encoder_month.joblib'))

    df = pd.read_csv(data_path)
    predictions_path = os.path.join(PATH, 'data', 'synthetic_data.csv')
    with open(predictions_path, 'w', encoding='UTF8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['DateTime', 'Total'])

        for index, row in df.iterrows():
            day = label_encoder_day.inverse_transform([[int(row.day)]])
            month = label_encoder_month.inverse_transform([[int(row.month)]]) 
            hour = label_encoder_hour.inverse_transform([[int(row.hour)]])
            minute = label_encoder_minute.inverse_transform([[int(row.minute)]])

            dt = str(day[0]) + '/' + str(month[0]) + '/2022 ' + str(hour[0]) + ':' + str(minute[0])
            writer.writerow([dt, future_preditions[index]])
            
    plt.title('Power production prediction')
    plt.ylabel('Solar energy production (kWh).')
    plt.grid(True)
    plt.autoscale(axis='x', tight=True)

    plt.plot(actual_results, label='Ground truth')
    plt.plot(future_preditions, label='Predictions')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()

