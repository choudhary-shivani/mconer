from dataloader import CoNLLReader
from tutils import *
import numpy as np
from NERmodel import NERmodelbase
from torch.utils.data import DataLoader
from tutils import invert, indvidual
from tutils import mconer_grouped


def collate_batch(batch):
    batch_ = list(zip(*batch))
    tokens, masks, token_masks, gold_spans, tags = batch_[0], batch_[1], batch_[2], batch_[3], batch_[4]

    max_len = np.max([len(token) for token in tokens])
    # print(np.max([len(token) for token in tokens]), max_len)
    token_tensor = torch.empty(size=(len(tokens), max_len), dtype=torch.long).fill_(tokenizer.pad_token_id)
    tag_tensor = torch.empty(size=(len(tokens), max_len), dtype=torch.long).fill_(mconern['O'])
    mask_tensor = torch.zeros(size=(len(tokens), max_len), dtype=torch.bool)
    token_masks_tensor = torch.zeros(size=(len(tokens), max_len), dtype=torch.bool)
    # print(token_tensor.shape)
    for i in range(len(tokens)):
        tokens_ = tokens[i]
        seq_len = len(tokens_)

        token_tensor[i, :seq_len] = tokens_
        tag_tensor[i, :seq_len] = tags[i]
        mask_tensor[i, :seq_len] = masks[i]
        token_masks_tensor[i, :seq_len] = token_masks[i]

    return token_tensor, tag_tensor, mask_tensor, token_masks_tensor, gold_spans


NUM_EPOCH = 10
BATCH_SIZE = 32
fine = True
mconern = indvidual(mconer_grouped, fine)
reveremap = invert(mconer_grouped)


ds = CoNLLReader(target_vocab=mconern, encoder_model=encoder_model, reversemap=reveremap, finegrained=fine)
ds.read_data(data=r'C:\Users\Rah12937\PycharmProjects\mconer\multiconer2023\train_dev\en-train.conll')

valid = CoNLLReader(target_vocab=mconern, encoder_model=encoder_model, reversemap=reveremap, finegrained=fine)
valid.read_data(data=r'C:\Users\Rah12937\PycharmProjects\mconer\multiconer2023\train_dev\en-dev.conll')

# test = CoNLLReader(target_vocab=mconern, encoder_model=encoder_model)
# test.read_data(data=r'C:\Users\Rah12937\PycharmProjects\mconern\multiconer2022\EN-English\en_dev.conll')

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

model = NERmodelbase(tag_to_id=mconern, device=device, encoder_model=encoder_model, dropout=0.3).to(device)
criterion = torch.nn.CrossEntropyLoss()


trainloader = DataLoader(ds, batch_size=BATCH_SIZE, collate_fn=collate_batch, num_workers=0, shuffle=True)
validloader = DataLoader(valid, batch_size=BATCH_SIZE, collate_fn=collate_batch, num_workers=0)
WARMUP_STEP = int(len(trainloader) * NUM_EPOCH * 0.1)
print(f"Number of warm up step is {WARMUP_STEP}")
optim, scheduler = get_optimizer(model, True, warmup=WARMUP_STEP)