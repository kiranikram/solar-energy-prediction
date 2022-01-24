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

    APPLICATION = 'transfomer_{}_{}'.format(args.epochs, MODEL_NUM_LAYERS)
    data_path = os.path.join(PATH, 'data', 'sunrock_clean_pe_april.csv')

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
    data_loader = DataLoader(train_dataset, **dataloader_params)

    model = Transformer(feature_size=MODEL_FEATURE_SIZE, num_layers=MODEL_NUM_LAYERS, dropout=0, emb_sizes=EMBEDDING_SIZES_TRANSFORMERS).double().to(DEVICE)

    model_path = os.path.join('saved_models', "{}.pth".format(APPLICATION))
    model.load_state_dict(torch.load(model_path))
    model = model.to(DEVICE)
    model.eval()

    #label_encoder_day = load(os.path.join(PATH, 'encoders', 'label_encoder_day.joblib'))
    #label_encoder_hour = load(os.path.join(PATH, 'encoders', 'label_encoder_hour.joblib'))
    #label_encoder_minute = load(os.path.join(PATH, 'encoders', 'label_encoder_minute.joblib'))
    #label_encoder_month = load(os.path.join(PATH, 'encoders', 'label_encoder_month.joblib'))


    df = pd.read_csv(data_path)
    df['DateTime'] = pd.to_datetime(df['DateTime'])

    df.iloc[0, df.columns.get_loc('Total')] = 0

    predictions_path = os.path.join(PATH, 'data', 'predictions.csv')
    with open(predictions_path, 'w', encoding='UTF8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        ptu = ['ptu_' + str(i + 1) for i in range(NO_OF_PTU)]
        ptu.insert(0, "DateTime")
        writer.writerow(ptu)

        for index, row in df.iterrows():

            ptu_preditions = []
            if(index == 0):
                y_1 = torch.tensor([row.Total]).double().to(DEVICE)

            # Get next batch (192 readings)
            next_ptus = df.iloc[index:index+NO_OF_PTU]
            
            for idx_ptu, row_ptu in next_ptus.iterrows():            
                X = torch.tensor([row_ptu.sin_day, row_ptu.cos_day, row_ptu.sin_month, row_ptu.cos_month, row_ptu.sin_hour, row_ptu.cos_hour, row_ptu.sin_min, row_ptu.cos_min, y_1])
                X = X.float().to(DEVICE)
                X_Emb = torch.tensor([row_ptu.day, row_ptu.month, row_ptu.hour, row_ptu.minute]).long().to(DEVICE)

                X= X.unsqueeze(0).unsqueeze(0)
                X_Emb = X_Emb.unsqueeze(0).unsqueeze(0)
                
                y_1 = model(X, X_Emb, DEVICE)

                ptu_preditions.append(y_1.item()  * 100)

                if(idx_ptu == 0 and index < len(df) - 1):
                    df.iloc[index + 1, df.columns.get_loc('Total')] = y_1.item()

            #day = label_encoder_day.inverse_transform([[int(row.day)]])
            #month = label_encoder_month.inverse_transform([[int(row.month)]]) 
            #hour = label_encoder_hour.inverse_transform([[int(row.hour)]])
            #minute = label_encoder_minute.inverse_transform([[int(row.minute)]])

            #dt = datetime.datetime(2021, month[0], day[0], hour[0], minute[0])

            ptu_preditions.insert(0,  row.DateTime.strftime('%Y/%m/%d %H:%M'))
            writer.writerow(ptu_preditions)

            logging.info('index {}, dt {}'.format( index, row.DateTime.strftime('%d/%m/%Y %H:%M')))


if __name__ == "__main__":
    main()

