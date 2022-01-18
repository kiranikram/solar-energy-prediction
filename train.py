import logging
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from dataset_solar import DatasetSolar
from model import LSTM

PATH  =  os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
lOG_LEVEL = 'DEBUG'
LOG_LEVEL = logging.getLevelName('DEBUG')
logging.basicConfig(level=LOG_LEVEL)

def train(model, epochs, optimizer, data_loader):

    start_time = time.time()
    criterion = nn.MSELoss()
    criterion = criterion.to(DEVICE)
    loss_list = []

    for epoch in range(epochs):
        logging.info('Epoch {}, lr {}'.format( epoch, optimizer.param_groups[0]['lr']))
        epoch_loss = 0
        total_iterations = 0

        for batch in data_loader:
            X, y = batch
            X = X.to(DEVICE)
            y = y.to(DEVICE).squeeze(0)

            #X = X.float()
            y = y.float()

            output = model(X) 

            output = output.squeeze(1)
            loss = criterion(output, y)

            epoch_loss += loss.clone().detach().cpu().numpy()
            total_iterations += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # average loss this epoch, divide by the number of batches in the dataset
        epoch_loss = epoch_loss/total_iterations
        loss_list.append(epoch_loss)
        logging.info("Loss in epoch {0:d} = {1:.3f}".format(epoch, epoch_loss))
    
    x =1


def main():

    data_path = os.path.join(PATH, 'data', 'sunrock_clean.csv')

    categorical_features = ["day", "month", "hour", "minute"]
    output_feature = "Total"

    sequence_length = 10
    batch_size = 1
    emb_sizes = [(31, 16), (12, 6), (24, 12), (4, 2)]

    learning_rate = 1e-3 
    weight_decay = 1e-4
    epochs = 10

    # parameters for the dataset
    dataset_params = {
        'data_path': data_path,
        'seq_length' : sequence_length,
        'device' : DEVICE,
        'categorical_features' : categorical_features,
        'output_feature' : output_feature
    }

    
    train_dataset = DatasetSolar(**dataset_params)
    dataloader_params = {'shuffle': False, 'batch_size': batch_size}
    data_loader = data.DataLoader(train_dataset, **dataloader_params)

    print(len(data_loader))

    #lstm_input_size, lstm_hidden_size, lstm_num_layers, seq_length, num_outputs
    #lstm_input_size, lstm_hidden_size, lstm_num_layers, h_fc_dim, seq_length, num_outputs
    
    model = LSTM(
        lstm_input_size=36,
        lstm_hidden_size=1024,
        lstm_num_layers=3,
        h_fc_dim=1024,
        emb_sizes=emb_sizes,
        seq_length=10,
        num_outputs=1
    )

    model = model.to(DEVICE)
    model.train()

    #count_parameters(model)

    logging.info(sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values()))

    optimizer_pars = {'lr': learning_rate, 'weight_decay': weight_decay}
    model_params = list(model.parameters())

    optimizer = torch.optim.Adam(model_params, **optimizer_pars)

    train(model, epochs, optimizer, data_loader)




if __name__ == "__main__":
    main()

