
import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):

    def __init__(self, lstm_input_size, lstm_hidden_size, lstm_num_layers, h_fc_dim, emb_sizes, seq_length, num_outputs):
        super(LSTM, self).__init__()
        
        self.num_outputs = num_outputs
        self.lstm_num_layers = lstm_num_layers
        self.lstm_input_size = lstm_input_size
        self.lstm_hidden_size = lstm_hidden_size
        self.seq_length = seq_length
        
        self.lstm = nn.LSTM(input_size=lstm_input_size, 
                            hidden_size=lstm_hidden_size, 
                            num_layers=lstm_num_layers, 
                            batch_first=True)

        self.day_emb = nn.Embedding(emb_sizes[0][0], emb_sizes[0][1])
        self.month_emb = nn.Embedding(emb_sizes[1][0], emb_sizes[1][1])
        self.hour_emb = nn.Embedding(emb_sizes[2][0], emb_sizes[2][1])
        self.minute_emb = nn.Embedding(emb_sizes[3][0], emb_sizes[3][1])
        
        self.fc1 = nn.Linear(lstm_hidden_size, lstm_hidden_size // 2)

        self.fc2 = nn.Linear(lstm_hidden_size // 2, num_outputs)

    def forward(self, x):
        xd = x[:, :, 0:1]
        xm = x[:, :, 1:2]
        xh = x[:, :, 2:3]
        xm = x[:, :, 3:4]

        x_day = self.day_emb(x[:, :, 0:1])
        x_month = self.month_emb(x[:, :, 1:2])
        x_hour = self.hour_emb(x[:, :, 2:3])
        x_minute = self.minute_emb(x[:, :, 3:4])



        #x = torch.cat([x_day, x_month, x_hour, x_minute])
        x = torch.cat([x_day, x_month, x_hour, x_minute], dim=3)
        x = x.squeeze(2)

        lstm_out, _ = self.lstm(x) 

        # Final hidden state
        #f_hs = output[:, -1, :]
        h_out = lstm_out.view(-1, self.lstm_hidden_size)


        # FC layers, batch first!
        x = self.fc1(h_out)
        x = F.relu(x)
        out = self.fc2(x)

       # h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
       # c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        # Propagate input through LSTM
       # ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        
       # h_out = h_out.view(-1, self.hidden_size)
        
       # out = self.fc(h_out)
        
        return out
