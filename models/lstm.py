
import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):

    def __init__(self, lstm_input_size, lstm_hidden_size, lstm_num_layers, emb_sizes, num_outputs, dropout_prob, device):
        super(LSTM, self).__init__()
        
        self.num_outputs = num_outputs
        self.lstm_num_layers = lstm_num_layers
        self.lstm_input_size = lstm_input_size
        self.lstm_hidden_size = lstm_hidden_size
        
        self.lstm = nn.LSTM(input_size=lstm_input_size, 
                            hidden_size=lstm_hidden_size, 
                            num_layers=lstm_num_layers, 
                            batch_first=True,
                            dropout=dropout_prob)

        self.day_emb = nn.Embedding(emb_sizes[0][0], emb_sizes[0][1])
        self.month_emb = nn.Embedding(emb_sizes[1][0], emb_sizes[1][1])
        self.hour_emb = nn.Embedding(emb_sizes[2][0], emb_sizes[2][1])
        self.minute_emb = nn.Embedding(emb_sizes[3][0], emb_sizes[3][1])
        
        self.fc1 = nn.Linear(lstm_hidden_size, lstm_hidden_size // 2)
        self.fc2 = nn.Linear(lstm_hidden_size // 2, num_outputs)
        self.device = device

    def forward(self, x, y):

        x_day = self.day_emb(x[:, :, 0:1])
        x_month = self.month_emb(x[:, :, 1:2])
        x_hour = self.hour_emb(x[:, :, 2:3])
        x_minute = self.minute_emb(x[:, :, 3:4])

        y = y.reshape(y.shape[0], y.shape[1], 1, 1)
        x = torch.cat([x_day, x_month, x_hour, x_minute, y], dim=3)
        x = x.squeeze(2)

        h0 = torch.zeros(self.lstm_num_layers, x.size(0), self.lstm_hidden_size).to(self.device).requires_grad_() #hidden state
        c0 = torch.zeros(self.lstm_num_layers, x.size(0), self.lstm_hidden_size).to(self.device).requires_grad_() #internal state
    

        lstm_out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach())) 

        out = lstm_out[:, -1, :]

        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        
        return out
