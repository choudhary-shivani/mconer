import sys
import numpy as np
import torch
from torch import nn
from torch.nn import LSTM, LSTMCell
from torch.optim import AdamW
from tensorboardX import SummaryWriter
from tqdm import tqdm
from pytorchtools import EarlyStopping
from lstmdl import trainloader, validloader, sd
from torchmetrics.classification import MulticlassConfusionMatrix, MulticlassRecall, MulticlassPrecision, MulticlassF1Score


class CustomLSTM(nn.Module):
    def __init__(self, input, hidden_size, num_layers=1, dropout=0.1):
        super(CustomLSTM, self).__init__()
        self.input = input
        self.hidden = hidden_size
        self.num_layers = num_layers
        self.embed = nn.Embedding(self.input, self.input, padding_idx=sd.num_char)
        self.lstm = LSTM(input_size=self.input,
                         hidden_size=self.hidden,
                         num_layers=self.num_layers,
                         batch_first=True,
                         dropout=dropout, bidirectional=True
                         )
        self.dropout = nn.Dropout(0.1)
        print(sd.tag_to_id)
        self.ff1 = nn.Linear(self.hidden*2, self.hidden)
        self.ff = nn.Linear(self.hidden, sd.num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        out = self.embed(x)
        self.h0 = torch.zeros(2*self.num_layers, batch_size, self.hidden, dtype=
        torch.float).to(device).requires_grad_()
        self.c0 = torch.zeros(2*self.num_layers, batch_size, self.hidden,
                              dtype=torch.float).to(device).requires_grad_()
        # lstm_out, _ = self.lstm(x.view(len(x), 1, -1))
        out , (self.h0, self.c0) = self.lstm(out, (self.h0, self.c0))
        # print(out.shape)
        h_comb = torch.cat([self.h0[-1], self.h0[-2]], dim=-1)
        # print(h_comb)
        # val = self.dropout(self.h0.mean(dim=0, keepdim=True))
        out = self.ff1(h_comb)
        out = self.ff(out)
        return out


device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
HIDDEN = 128
DROP_OUT = 0.2
NUM_EPOCH = 10
cps = 1500
model = CustomLSTM(sd.num_char+1, HIDDEN, num_layers=1, dropout=DROP_OUT).to(device)
optim = AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
import random

early_stopping = EarlyStopping(patience=10, verbose=True, path='lstm_model.pt')
num = random.randint(1, 100)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.95)
runid = random.randint(1, 100000)
writer = SummaryWriter(f"Run_{runid}_EP_{NUM_EPOCH}_hid_{HIDDEN}_dr_{DROP_OUT}_num_{num}")
print(runid)
confmat = MulticlassConfusionMatrix(num_classes=sd.num_classes)
pred = MulticlassPrecision(num_classes=sd.num_classes, average=None)
recall = MulticlassRecall(num_classes=sd.num_classes, average=None)
f1 = MulticlassF1Score(num_classes=sd.num_classes, average=None)
print(model)
# sys.exit(2)
step = 0
for epoch in range(NUM_EPOCH):
    val_loss_all = []
    with tqdm(trainloader, unit='batch') as tepoch:
        tepoch.set_description(f"Epoch {epoch}")
        running_loss = 0
        for i, data in enumerate(tepoch):
            optim.zero_grad()
            x, y = data
            # output = model(x.view(x.size(0), -1, 1).to(device))
            output = model(x.to(device))
            x.cpu()
            # print(output.shape)
            loss = criterion(output.cpu(), torch.tensor(y))
            # print(loss)
            running_loss += loss
            loss.backward()
            optim.step()
            step += 1

            # if ((step + 1) % cps) == 0:
                # # Calculate validation loss
    print("Validating the Model")
    # with tqdm(validloader, unit='batch') as tepoch:
        # tepoch.set_description(f"Valid Epoch {epoch}")
    model.eval()
    with torch.no_grad():
        valid_loss = 0
        out_all =[]
        y_all = []
        for i, data in enumerate(validloader):
            x, y = data
            y = torch.tensor(y)
            # output = model(x.view(x.size(0), -1, 1).to(device))
            output = model(x.to(device))
            x.cpu()
            # print(output.shape, y.shape)
            y_all.append(y)
            out_all.append(output.cpu())
            valid_loss += criterion(output.cpu(), y)
    print(confmat(torch.argmax(torch.softmax(torch.cat(out_all), dim=1), dim=1),
                           torch.cat(y_all)),
          pred(torch.argmax(torch.softmax(torch.cat(out_all), dim=1), dim=1),
                           torch.cat(y_all)),
          recall(torch.argmax(torch.softmax(torch.cat(out_all), dim=1), dim=1),
                           torch.cat(y_all)),
          f1(torch.argmax(torch.softmax(torch.cat(out_all), dim=1), dim=1),
           torch.cat(y_all))
          )
    # print()
    model.train()
    print('Loss/train', running_loss / len(trainloader), epoch + 1)
    print('Loss/valid', valid_loss / len(validloader), epoch + 1)
    writer.add_scalars('Loss', {"train": running_loss / cps,
                                "valid": valid_loss / len(validloader)}, step + 1)
    val_loss_all.append((valid_loss / len(validloader)))
                # writer.add_scalar('Loss/valid', valid_loss / len(validloader), step + 1)
    early_stopping(np.mean(val_loss_all), model)
    if early_stopping.early_stop:
        print("Stopping early")
        break
    # writer.add_scalar('Loss/train', running_loss / len(train_data), epoch + 1)
    # writer.add_scalar('Loss/valid', valid_loss / len(valid_data), epoch + 1)
    # print(f"\nAfter Epoch {epoch} validation loss is {valid_loss / len(valid_data):.5f} and "
    #       f"Train loss: {running_loss / len(train_data):.5f}\n")
writer.close()
