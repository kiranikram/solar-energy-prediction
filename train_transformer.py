import argparse
import datetime
import json
import logging
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import  DataLoader
from datasets.dataset_transformer import DatasetTransformer
from models.transformer import Transformer
from lib.constants import *
from lib import plot_utils

PATH  =  os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
LOG_LEVEL = logging.getLevelName('DEBUG')
logging.basicConfig(level=LOG_LEVEL)

def get_args_parser():
    parser = argparse.ArgumentParser('Solar panel production prediction', add_help=False)
    parser.add_argument('--env', type=str, default="laptop", help='Enviroment [default: laptop]')
    parser.add_argument('--lr', default=0.0001 , type=float)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--seq_length', default=10, type=int)
    
    return parser.parse_args()

def train(model, epochs, optimizer, data_loader, data_loader_val, APPLICATION):
    
    start_time = time.time()

    logging.info("Start training {}".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M')))
    criterion = nn.MSELoss()
    criterion = criterion.to(DEVICE)
    loss_list_train = []
    loss_list_val = []

    for epoch in range(epochs):
        model.train()
        logging.info('Epoch {}, lr {}'.format( epoch, optimizer.param_groups[0]['lr']))

        epoch_loss_train = 0
        epoch_loss_val = 0
        total_iterations = 0
        total_iterations_val = 0

        idx_train = 0
        for batch in data_loader:
            optimizer.zero_grad()

            X, X_Emb, y = batch
            X = X.to(DEVICE).float()
            X_Emb = X_Emb.to(DEVICE)
            y = y.to(DEVICE).double()

            prediction = model(X, X_Emb, DEVICE)
            prediction = prediction.reshape(-1, 1)
            y = y.reshape(-1, 1)
            loss = criterion(prediction, y)
            l = loss.item()
            epoch_loss_train += l
            total_iterations += 1
            
            loss.backward()
            optimizer.step()

            if idx_train%100 == 1:
                logging.info(f'iteration: {idx_train:3} loss: {l:10.8f}')

            idx_train += 1
        
        # average loss this epoch, divide by the number of batches in the dataset
        epoch_loss_train = epoch_loss_train/total_iterations
        loss_list_train.append(epoch_loss_train)
        logging.info("Train loss in epoch {0:d} = {1:.3f}".format(epoch, epoch_loss_train))

        #GET AVERAGE VALIDATION LOSS
        model.eval()
        idx_val = 0
        for batch in data_loader_val:
            with torch.no_grad():
                X, X_Emb, y = batch
                X = X.to(DEVICE).float()
                X_Emb = X_Emb.to(DEVICE)
                y = y.to(DEVICE).double()

                prediction = model(X, X_Emb, DEVICE)
                prediction = prediction.reshape(-1, 1)
                y = y.reshape(-1, 1)
                val_loss = criterion(prediction, y)
                l_v = val_loss.item()
                epoch_loss_val += l_v
                total_iterations_val += 1

                if idx_val%100 == 1:
                    logging.info(f'iteration: {idx_val:3} Val loss: {val_loss:10.8f}')
            idx_val += 1
        # average loss this epoch, divide by the number of batches in the dataset
        epoch_loss_val = epoch_loss_val/total_iterations_val
        loss_list_val.append(epoch_loss_val)
        logging.info("Validation loss in epoch {0:d} = {1:.3f}".format(epoch, epoch_loss_val))
        

    end_time = time.time()

    logging.info("End training {}".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M')))
    logging.info("Training took {0:.1f}".format(end_time-start_time))

    # visualization loss
    path = os.path.join('plots', '{}_{}_loss_train.png'.format(APPLICATION, epochs))
    plot_utils.save_plot(loss_list_train, 'Loss', 'epochs', "Training loss", path)
    f = open(path.replace(".png", ".json"), 'w')
    json.dump(loss_list_train, f)
    f.close()

    path = os.path.join('plots', '{}_{}_loss_val.png'.format(APPLICATION, epochs))
    plot_utils.save_plot(loss_list_val, 'Loss', 'epochs', "Validation loss", path)
    f = open(path.replace(".png", ".json"), 'w')
    json.dump(loss_list_val, f)
    f.close()

    model = model.to(torch.device('cpu'))
    model_path = os.path.join('saved_models', "{}.pth".format(APPLICATION))
    torch.save(model.state_dict(), model_path)

def main():

    args = get_args_parser()
    APPLICATION = 'transfomer_{}_{}'.format(args.epochs, MODEL_NUM_LAYERS)
    
    data_path_train = os.path.join(PATH, 'data', 'sunrock_clean_spring_train.csv')
    data_path_val = os.path.join(PATH, 'data', 'sunrock_clean_spring_val.csv')

    # parameters for the dataset
    dataset_params = {
        'data_path': data_path_train,
        'seq_length' : args.seq_length,
        'device' : DEVICE,
        'features' : TRANSFOMERS_FEATURES,
        'categorical_features' : CATEGORICAL_FEATURES,
        'output_feature' : OUTPUT_FEATURE
    }
    
    dataset_train = DatasetTransformer(**dataset_params)
    dataloader_params = {'shuffle': False, 'batch_size': args.batch_size}
    data_loader_train = DataLoader(dataset_train, **dataloader_params)

    dataset_params = {
        'data_path': data_path_val,
        'seq_length' : args.seq_length,
        'device' : DEVICE,
        'features' : TRANSFOMERS_FEATURES,
        'categorical_features' : CATEGORICAL_FEATURES,
        'output_feature' : OUTPUT_FEATURE
    }
    
    dataset_val = DatasetTransformer(**dataset_params)
    dataloader_params = {'shuffle': False, 'batch_size': args.batch_size}
    data_loader_val = DataLoader(dataset_val, **dataloader_params)

    model = Transformer(feature_size=MODEL_FEATURE_SIZE, num_layers=MODEL_NUM_LAYERS, dropout=0, emb_sizes=EMBEDDING_SIZES_TRANSFORMERS).double().to(DEVICE)
    optimizer_pars = {'lr': args.lr}
    model_params = list(model.parameters())
    optimizer = torch.optim.Adam(model_params, **optimizer_pars)

    train(model, args.epochs, optimizer, data_loader_train, data_loader_val, APPLICATION)

if __name__ == "__main__":
    main()
