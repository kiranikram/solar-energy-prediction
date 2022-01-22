import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data

from models.lstm import FeedForwardNN
from tabular_dataset import TabularDataset

PATH  =  os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def saveModel(model): 
    path = "NetModel.pth" 
    torch.save(model.state_dict(), path) 

def main():

    df = pd.read_csv(os.path.join(PATH, '..\\' 'data', 'sunrock_raw.csv'), header=0)

    df['DateTime'] = pd.to_datetime(df['DateTime'])

    df['day'] = df['DateTime'].dt.day
    df['month'] = df['DateTime'].dt.month
    df['hour'] = df['DateTime'].dt.hour
    df['minute'] = df['DateTime'].dt.minute

    # Drop DateTime column
    df = df.drop(columns=['DateTime'])

    categorical_features = ["day", "month", "hour", "minute"]
    output_feature = "Total"

    from sklearn.preprocessing import LabelEncoder
    label_encoders = {}
    for cat_col in categorical_features:
        label_encoders[cat_col] = LabelEncoder()
        print(df[cat_col] )
        df[cat_col] = label_encoders[cat_col].fit_transform(df[cat_col])
        print(df[cat_col] )

    dataset = TabularDataset(data=df, cat_cols=categorical_features, output_col=output_feature)

    batchsize = 64
    dataloader = data.DataLoader(dataset, batchsize, shuffle=True, num_workers=1)

    cat_dims = [int(df[col].nunique()) for col in categorical_features]

    emb_dims = [(x, min(50, (x + 1) // 2)) for x in cat_dims]

    model = FeedForwardNN(emb_dims, no_of_cont=0, lin_layer_sizes=[50, 100], output_size=1, emb_dropout=0.04, lin_layer_dropouts=[0.001,0.01]).to(DEVICE)

    no_of_epochs = 100
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    for epoch in range(no_of_epochs):
        running_train_loss = 0.0 
        for y, cont_x, cat_x in dataloader:
            
            cat_x = cat_x.to(DEVICE)
            cont_x = cont_x.to(DEVICE)
            y  = y.to(DEVICE)

            # Forward Pass
            preds = model(cont_x, cat_x)
            loss = criterion(preds, y)

            # Backward Pass and Optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()

        # Calculate training loss value 
        train_loss_value = running_train_loss/len(dataloader)

        print("Epoch, loss: {} / {}".format(epoch+1, train_loss_value)) 

    saveModel(model)

if __name__ == "__main__":
    main()
