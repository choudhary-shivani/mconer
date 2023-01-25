import torch
import numpy as np
from torch.utils.data import DataLoader
from dataprep import Serialdata


def collate_batch(batch):
    batch_ = list(zip(*batch))
    tokens, label = batch_[0], batch_[1]

    max_len = np.max([len(token) for token in tokens])
    # print(np.max([len(token) for token in tokens]), max_len)
    token_tensor = torch.empty(size=(len(tokens), max_len), dtype=torch.long).fill_(sd.num_char)

    # print(token_tensor.shape)
    for i in range(len(tokens)):
        tokens_ = tokens[i]
        seq_len = len(tokens_)
        token_tensor[i, 0:seq_len] = tokens_

    return token_tensor, label


sd = Serialdata('combined.tsv')
sd.readLines(train=True)

vd = Serialdata('combined.tsv')
vd.readLines(train=False)

trainloader = DataLoader(sd, batch_size=128, collate_fn=collate_batch, shuffle=True)
validloader = DataLoader(vd, batch_size=256, collate_fn=collate_batch, shuffle=True)

