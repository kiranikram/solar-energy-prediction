import torch.nn as nn
import torch, math
import time
import torch.nn.functional as F

class Transformer(nn.Module):
    # d_model : number of features
    def __init__(self,feature_size=7,num_layers=3,dropout=0, emb_sizes=[]):
        super(Transformer, self).__init__()

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=4, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)        
        self.decoder = nn.Linear(feature_size,1)
        self.init_weights()

        self.day_emb = nn.Embedding(emb_sizes[0][0], emb_sizes[0][1])
        self.month_emb = nn.Embedding(emb_sizes[1][0], emb_sizes[1][1])
        self.hour_emb = nn.Embedding(emb_sizes[2][0], emb_sizes[2][1])
        self.minute_emb = nn.Embedding(emb_sizes[3][0], emb_sizes[3][1])

    def init_weights(self):
        initrange = 0.1    
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x, x_emb, device):

        x_day = self.day_emb(x_emb[:, :, 0:1])
        x_month = self.month_emb(x_emb[:, :, 1:2])
        x_hour = self.hour_emb(x_emb[:, :, 2:3])
        x_minute = self.minute_emb(x_emb[:, :, 3:4])

        x = x.unsqueeze(2)
        #print(x.shape)
        x = torch.cat([x, x_day, x_month, x_hour, x_minute], dim=3)
        x = x.squeeze(2)
        
        mask = self._generate_square_subsequent_mask(len(x)).to(device)
        output = self.transformer_encoder(x,mask)
        output = self.decoder(output)

        return output