import argparse
import json
import logging
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import  DataLoader
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
    parser.add_argument('--lr', default=0.0001 , type=float)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--seq_length', default=24, type=int)
    parser.add_argument('--lstm_hidden_size', default=1024, type=int)
    parser.add_argument('--lstm_num_layers', default=5, type=int)

    return parser.parse_args()

def train(model, epochs, optimizer, data_loader, APPLICATION):

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

            X, y, y_t1 = batch
            X = X.to(DEVICE)
            y = y.to(DEVICE)
            y = y.float()
            
            y_t1 = y_t1.to(DEVICE)
            y_t1 = y_t1.float()


            output = model(X, y) 

            output = output.squeeze(1)

            loss = criterion(output, y_t1)
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

    APPLICATION = 'lstm_{}_{}'.format(args.lstm_hidden_size, args.lstm_num_layers)

    data_path = os.path.join(PATH, 'data', 'sunrock_clean.csv')

    categorical_features = ["day", "month", "hour", "minute"]
    output_feature = "Total"

    emb_sizes = [(31, 16), (12, 6), (24, 12), (4, 2)]

    learning_rate = args.lr 
    weight_decay = 1e-4
    epochs = args.epochs

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
    data_loader = DataLoader(train_dataset, **dataloader_params)

    model = LSTM(
        lstm_input_size=37,
        lstm_hidden_size=args.lstm_hidden_size,
        lstm_num_layers=args.lstm_num_layers,
        emb_sizes=emb_sizes,
        num_outputs=1,
        dropout_prob=0.1,
        device=DEVICE
    )

    model = model.to(DEVICE)
    model.train()

    logging.info(APPLICATION)

    optimizer_pars = {'lr': learning_rate}
    model_params = list(model.parameters())

    optimizer = torch.optim.Adam(model_params, **optimizer_pars)

    train(model, epochs, optimizer, data_loader, APPLICATION)

if __name__ == "__main__":
    main()

