import os

import numpy as np
import sys

import pandas as pd
import torch
import random
from transformers import get_constant_schedule_with_warmup
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from NERmodel import NERmodelbase
from dataloader import CoNLLReader
from pytorchtools import EarlyStopping
from torch.optim.lr_scheduler import ReduceLROnPlateau
# from torch.utils.tensorboard import SummaryWriter
from utils import *
from tensorboardX import SummaryWriter
# from utils import mconern, encoder_model, tokenizer
from reader import *
# from tutils import image_gen
from tqdm import tqdm
from kornia.losses import FocalLoss
from torchmetrics.classification import MulticlassConfusionMatrix, MulticlassRecall, MulticlassPrecision, MulticlassF1Score

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

# sys.exit(0)
mconern = indvidual(mconer_grouped, True)
reveremap = invert(mconer_grouped)


def collate_batch(batch):
    batch_ = list(zip(*batch))
    tokens, masks, token_masks, gold_spans, tags, lstm_encoded = batch_[0], batch_[1], batch_[2], batch_[3], \
                                                                 batch_[4], batch_[5]
    # print(tags)
    max_len = np.max([len(token) for token in tokens])
    # print(np.max([len(token) for token in tokens]), max_len)
    token_tensor = torch.empty(size=(len(tokens), max_len), dtype=torch.long).fill_(tokenizer.pad_token_id)
    tag_tensor = torch.empty(size=(len(tokens), max_len), dtype=torch.long).fill_(mconern['O'])
    mask_tensor = torch.zeros(size=(len(tokens), max_len), dtype=torch.bool)
    token_masks_tensor = torch.zeros(size=(len(tokens), max_len), dtype=torch.bool)
    lstm_encoded_tensor = torch.zeros(size=(len(tokens), max_len, 256), dtype=torch.float)
    # print(lstm_encoded.shape)
    for i in range(len(tokens)):
        tokens_ = tokens[i]
        seq_len = len(tokens_)

        token_tensor[i, :seq_len] = tokens_
        tag_tensor[i, :seq_len] = tags[i]
        mask_tensor[i, :seq_len] = masks[i]
        token_masks_tensor[i, :seq_len] = token_masks[i]
        lstm_encoded_tensor[i, 1:seq_len - 1, :] = lstm_encoded[i]

    return token_tensor, tag_tensor, mask_tensor, token_masks_tensor, gold_spans, lstm_encoded_tensor


def trainer(NUM_EPOCH, BATCH_SIZE, fine, force, eval_step=100, ratio=0.2):

    if os.path.exists('train_load.pkl') and not force:
        with open('train_load.pkl', 'rb') as f:
            ds = pickle.load(f)
    else:
        print("reading from disk")
        ds = CoNLLReader(target_vocab=mconern, encoder_model=encoder_model, reversemap=reveremap, finegrained=fine)
        ds.read_data(data=r'C:\Users\Rah12937\PycharmProjects\mconer\multiconer2023\train_dev\en-train.conll')
        with open('train_load.pkl', 'wb') as f:
            pickle.dump(ds, f)

    if os.path.exists('valid_load.pkl') and not force:
        with open('valid_load.pkl', 'rb') as f:
            valid = pickle.load(f)
    else:
        valid = CoNLLReader(target_vocab=mconern, encoder_model=encoder_model, reversemap=reveremap, finegrained=fine)
        valid.read_data(data=r'C:\Users\Rah12937\PycharmProjects\mconer\multiconer2023\train_dev\en-dev.conll')
        with open('valid_load.pkl', 'wb') as f:
            pickle.dump(valid, f)
    # confmat = MulticlassConfusionMatrix(num_classes=len(mconern))
    # pred = MulticlassPrecision(num_classes=len(mconern), average=None)
    # recall = MulticlassRecall(num_classes=len(mconern), average=None)
    # f1 = MulticlassF1Score(num_classes=len(mconern), average=None)
    # test = CoNLLReader(target_vocab=mconern, encoder_model=encoder_model)
    # test.read_data(data=r'C:\Users\Rah12937\PycharmProjects\mconern\multiconer2022\EN-English\en_dev.conll')

    # model = NERmodelbase(tag_to_id=mconern, device=device, encoder_model=encoder_model,
    #                      dropout=0.3, use_lstm=False).to(device)
    model = NERmodelbase(tag_to_id=mconern, device=device, encoder_model=encoder_model, dropout=0.3, use_lstm=True).to(
        device)
    print(model)
    trainloader = DataLoader(ds, batch_size=BATCH_SIZE, collate_fn=collate_batch, num_workers=0, shuffle=True)
    validloader = DataLoader(valid, batch_size=BATCH_SIZE, collate_fn=collate_batch, num_workers=0)
    WARMUP_STEP = int(len(trainloader) * NUM_EPOCH * 0.2) // 20
    print(f"Number of warm up step is {WARMUP_STEP}")
    optim, scheduler = get_optimizer(model, True, warmup=WARMUP_STEP)
    scheduler2 = ReduceLROnPlateau(optimizer=optim[0], factor=0.01, patience=20, verbose=True, cooldown=20)

    run_id = random.randint(1, 10000)
    run_name = f"runid_{run_id}_EP_{NUM_EPOCH}_fine_xlm-b-birnnn-focal-loss-{0.8}-sep_lr-alpha-2-gama-4"
    while True:
        if os.path.exists(run_name):
            run_id = random.randint(1, 10000)
            run_name = f"runid_{run_id}_EP_{NUM_EPOCH}_fine_xlm-b-birnnn-focal-loss-{0.8}-sep_lr-alpha-2-gama-4"
        else:
            break
    os.mkdir(run_name)
    os.chdir(run_name)
    writer = SummaryWriter(run_name)
    step = 0
    running_loss = 0
    early_stopping = EarlyStopping(patience=10, verbose=True, path=run_name + '.pt')
    for epoch in range(NUM_EPOCH):
        val_track = []
        with tqdm(trainloader, unit='batch') as tepoch:
            # model.train()
            tepoch.set_description(f"Epoch {epoch}")
            for i, data in enumerate(tepoch):
                optim[0].zero_grad()
                outputs, focal_loss = model(data)
                loss = 0.2 * outputs['loss'] + 0.8 * focal_loss
                running_loss += loss
                loss.backward()
                optim[0].step()
                if (step +1 ) % 20 == 0:
                    scheduler[0].step()
                scheduler2.step(loss)
                # if i % 10 == 0:  # print every 2000 mini-batches
                model.spanf1.reset()
                # writer.add_scalar('lr', scheduler2.get_last_lr()[0], step)
                writer.add_scalar('lr', scheduler2._last_lr[0], step)
                # run validation
                step += 1
                if (step + 1) % eval_step == 0:
                    # all_tags = []
                    # all_predicted_tags = []
                    model.eval()
                    with torch.no_grad():
                        with tqdm(validloader, unit='batch') as tepoch:
                            val_loss = 0
                            for i, data in enumerate(tepoch):
                                # tokens, tags, mask, token_mask, metadata, lstm_encoded = data
                                # all_tags += list(torch.masked_select(tags, mask).detach().cpu().numpy())
                                # print(mask.shape, token_mask.shape)
                                # all_tags += list(tags.clone().detach().cpu().numpy().ravel())
                                outputs, focal_loss = model(data, mode='predict')
                                # val_loss += 0.6 * outputs['loss'] + 0.4 * focal_loss
                                val_loss += (0.2 * outputs['loss'] + 0.8 * focal_loss)
                                # val_loss = closs
                    model.train()
                    writer.add_scalars("Loss",
                                       {
                                           "Train Loss": round(running_loss.detach().cpu().numpy().ravel()[0] / 20, 4),
                                           "Valid Loss": round(val_loss.detach().cpu().numpy().ravel()[0] / 20, 4),
                                       }
                                       , step)
                    print(outputs['results'])
                    writer.add_scalars("Metrics", outputs['results'], step)
                    val_track.append(round(val_loss.detach().cpu().numpy().ravel()[0] / eval_step, 4))
                    running_loss = 0

                    early_stopping(round(val_loss.detach().cpu().numpy().ravel()[0] / eval_step, 4), model)
                    if early_stopping.early_stop:
                        print("Stopping early")
                        writer.close()
                        os.chdir('..')
                        return
                    # conf_mat = (confmat(
                    #     torch.tensor(all_tags), torch.tensor(all_predicted_tags)
                    # ))
                    # mat = pd.DataFrame(conf_mat.numpy(), columns=mconern.keys(), dtype=int)
                    # mat["idx"] = mconern.keys()
                    # mat.to_csv(f'confusion_mat{step}.csv', index=False)
    writer.close()
    os.chdir('..')

