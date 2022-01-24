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
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--seq_length', default=1, type=int)

    return parser.parse_args()
    
def main():

    args = get_args_parser()

    APPLICATION = 'transfomer_{}_{}'.format(args.epochs, MODEL_NUM_LAYERS)
    data_path = os.path.join(PATH, 'data', 'sunrock_clean_spring_test.csv')

    model = Transformer(feature_size=MODEL_FEATURE_SIZE, num_layers=MODEL_NUM_LAYERS, dropout=0, emb_sizes=EMBEDDING_SIZES_TRANSFORMERS).double().to(DEVICE)

    model_path = os.path.join('saved_models', "{}.pth".format(APPLICATION))
    model.load_state_dict(torch.load(model_path))
    model = model.to(DEVICE)
    model.eval()


    dataset_params = {
        'data_path': data_path,
        'seq_length' : args.seq_length,
        'device' : DEVICE,
        'features' : TRANSFOMERS_FEATURES,
        'categorical_features' : CATEGORICAL_FEATURES,
        'output_feature' : OUTPUT_FEATURE
    }
    
    dataset = DatasetTransformer(**dataset_params)
    dataloader_params = {'shuffle': False, 'batch_size': args.batch_size}
    data_loader = DataLoader(dataset, **dataloader_params)
    dataiter = iter(data_loader)

    for i in range(1):
        X, X_Emb, y = dataiter.next()
        
        X = X.to(DEVICE).float()
        X_Emb = X_Emb.to(DEVICE)
        y = y.to(DEVICE).double()

        prediction = model(X, X_Emb, DEVICE)

        print(prediction.item())

        x = 1

  
if __name__ == "__main__":
    main()

