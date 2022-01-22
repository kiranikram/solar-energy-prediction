import argparse
import json
import logging
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import  DataLoader
from datasets.dataset_transformer import DatasetTransformer
from models.transformer import Transformer
from lib import plot_utils

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
    parser.add_argument('--seq_length', default=20, type=int)
    parser.add_argument('--num_layers', default=4, type=int)
    
    return parser.parse_args()

def train(model, epochs, optimizer, data_loader, APPLICATION):
    model.train()
    start_time = time.time()
    criterion = nn.MSELoss()
    criterion = criterion.to(DEVICE)
    loss_list = []

    for epoch in range(epochs):
        logging.info('Epoch {}, lr {}'.format( epoch, optimizer.param_groups[0]['lr']))
        epoch_loss = 0
        total_iterations = 0

        i = 0
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
            epoch_loss += l
            total_iterations += 1
            
            loss.backward()
            optimizer.step()

            if i%100 == 1:
                logging.info(f'iteration: {i:3} loss: {l:10.8f}')

            i += 1
        
        # average loss this epoch, divide by the number of batches in the dataset
        epoch_loss = epoch_loss/total_iterations
        loss_list.append(epoch_loss)
        logging.info("Loss in epoch {0:d} = {1:.3f}".format(epoch, epoch_loss))

    end_time = time.time()

    logging.info("Training took {0:.1f}".format(end_time-start_time))

    # visualization loss
    path = os.path.join('plots', '{}_{}_loss.png'.format(APPLICATION, epochs))
    plot_utils.save_plot(loss_list, 'Training loss', 'epochs', "Training loss", path)
    f = open(path.replace(".png", ".json"), 'w')
    json.dump(loss_list, f)
    f.close()

    model = model.to(torch.device('cpu'))
    model_path = os.path.join('saved_models', "{}_{}.pth".format(APPLICATION, epochs))
    torch.save(model.state_dict(), model_path)

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
    data_loader = DataLoader(train_dataset, **dataloader_params)

    model = Transformer(feature_size=44, num_layers=args.num_layers, dropout=0, emb_sizes=emb_sizes).double().to(DEVICE)

    logging.info(APPLICATION)

    optimizer_pars = {'lr': learning_rate}
    model_params = list(model.parameters())

    optimizer = torch.optim.Adam(model_params, **optimizer_pars)

    train(model, epochs, optimizer, data_loader, APPLICATION)

if __name__ == "__main__":
    main()
