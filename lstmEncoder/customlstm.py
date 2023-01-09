import torch
from torch import nn
from torch.nn import LSTM
# from lstmEncoder.dataprep import Serialdata
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

# sd = Serialdata('combined.tsv')


class CustomLSTM(nn.Module):
    def __init__(self, input, hidden_size, num_layers=1, dropout=0.1):
        super(CustomLSTM, self).__init__()
        self.input = input
        self.hidden = hidden_size
        self.num_layers = num_layers
        self.embed = nn.Embedding(self.input, self.input, padding_idx=57)
        self.lstm = LSTM(input_size=self.input,
                         hidden_size=self.hidden,
                         num_layers=self.num_layers,
                         batch_first=True,
                         dropout=dropout
                         )
        self.dropout = nn.Dropout(0.1)
        # print(sd.tag_to_id)
        self.ff1 = nn.Linear(self.hidden, self.hidden//2)
        self.ff = nn.Linear(self.hidden//2, 5)

    def forward(self, x):
        batch_size = x.size(0)
        out = self.embed(x.to(device))
        self.h0 = torch.zeros(self.num_layers, batch_size, self.hidden, dtype=
        torch.float).to(device).requires_grad_()
        self.c0 = torch.zeros(self.num_layers, batch_size, self.hidden,
                              dtype=torch.float).to(device).requires_grad_()
        # lstm_out, _ = self.lstm(x.view(len(x), 1, -1))
        _, (self.h0, self.c0) = self.lstm(out, (self.h0, self.c0))
        # print(self.h0.shape)
        # val = self.dropout(self.h0.mean(dim=0, keepdim=True))
        out = self.ff1(self.h0[-1])
        out = self.ff(out)
        return out, self.h0