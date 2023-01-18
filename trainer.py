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
# from torch.utils.tensorboard import SummaryWriter
from utils import *
from tensorboardX import SummaryWriter
# from utils import mconern, encoder_model, tokenizer
from reader import *
from tutils import image_gen
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


def trainer(NUM_EPOCH, BATCH_SIZE, fine, force, eval_step=26, ratio=0.2):
    print(force)
    if os.path.exists('train_load.pkl') and not force:
        with open('valid_load.pkl', 'rb') as f:
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
    confmat = MulticlassConfusionMatrix(num_classes=len(mconern))
    pred = MulticlassPrecision(num_classes=len(mconern), average=None)
    recall = MulticlassRecall(num_classes=len(mconern), average=None)
    f1 = MulticlassF1Score(num_classes=len(mconern), average=None)
    # test = CoNLLReader(target_vocab=mconern, encoder_model=encoder_model)
    # test.read_data(data=r'C:\Users\Rah12937\PycharmProjects\mconern\multiconer2022\EN-English\en_dev.conll')

    # model = NERmodelbase(tag_to_id=mconern, device=device, encoder_model=encoder_model,
    #                      dropout=0.3, use_lstm=False).to(device)
    model = NERmodelbase(tag_to_id=mconern, device=device, encoder_model=encoder_model, dropout=0.3, use_lstm=True).to(
        device)
    print(model)
    trainloader = DataLoader(ds, batch_size=BATCH_SIZE, collate_fn=collate_batch, num_workers=0, shuffle=True)
    validloader = DataLoader(valid, batch_size=BATCH_SIZE, collate_fn=collate_batch, num_workers=0)
    WARMUP_STEP = int(len(trainloader) * NUM_EPOCH * 0.2)
    print(f"Number of warm up step is {WARMUP_STEP}")
    optim, scheduler = get_optimizer(model, True, warmup=WARMUP_STEP)

    run_id = random.randint(1, 10000)
    run_name = f"runid_{run_id}_EP_{NUM_EPOCH}_fine_xlm-b-birnnn-focal-loss-{1-ratio}-sep_lr-alpha-2-gama-4"
    writer = SummaryWriter(run_name)
    step = 0
    running_loss = 0
    early_stopping = EarlyStopping(patience=10, verbose=True, path=run_name + '.pt')
    # eval_step = 100
    # fl = FocalLoss()
    # print(next(iter(trainloader)))
    # sys.exit(2)
    for epoch in range(NUM_EPOCH):
        val_track = []
        with tqdm(trainloader, unit='batch') as tepoch:
            # model.train()
            tepoch.set_description(f"Epoch {epoch}")
            for i, data in enumerate(tepoch):
                optim[0].zero_grad()
                outputs, focal_loss = model(data)
                loss = ratio * outputs['loss'] + (1-ratio) * focal_loss
                running_loss += loss
                loss.backward()
                optim[0].step()
                scheduler[0].step()
                # if i % 10 == 0:  # print every 2000 mini-batches
                model.spanf1.reset()
                # writer.add_scalar('lr', scheduler[0].get_last_lr()[0], step)
                # run validation
                step += 1
                if (step + 1) % eval_step == 0:
                    all_tags = []
                    all_predicted_tags = []
                    model.eval()
                    with torch.no_grad():
                        with tqdm(validloader, unit='batch') as tepoch:
                            val_loss = 0
                            for i, data in enumerate(tepoch):
                                tokens, tags, mask, token_mask, metadata, lstm_encoded = data
                                all_tags += list(torch.masked_select(tags, mask).detach().cpu().numpy())
                                # print(mask.shape, token_mask.shape)
                                # all_tags += list(tags.clone().detach().cpu().numpy().ravel())
                                outputs, focal_loss = model(data, mode='predict')
                                for i in outputs['best_path']:
                                    all_predicted_tags += i[0]
                                # all_predicted_tags += [i[0] for i in outputs['best_path']]
                                val_loss += ratio * outputs['loss'] + (1-ratio) * focal_loss
                    model.train()
                    writer.add_scalars("Loss",
                                       {
                                           "Train Loss": round(running_loss.detach().cpu().numpy().ravel()[0] / 20, 4),
                                           "Valid Loss": round(val_loss.detach().cpu().numpy().ravel()[0] / 20, 4),
                                       }
                                       , step)
                    writer.add_scalars("Metrics", outputs['results'], step)
                    val_track.append(round(val_loss.detach().cpu().numpy().ravel()[0] / eval_step, 4))
                    running_loss = 0

                    early_stopping(round(val_loss.detach().cpu().numpy().ravel()[0] / eval_step, 4), model)
                    if early_stopping.early_stop:
                        print("Stopping early")
                        writer.close()
                        return
                    conf_mat = (confmat(
                        torch.tensor(all_tags), torch.tensor(all_predicted_tags)
                        # np.array(all_predicted_tags).ravel()
                    ))
                    # im = image_gen(conf_mat, mconern)
                    # writer.add_image("val_confusion_matrix", im, global_step=step)
                    pd.DataFrame(conf_mat.numpy(), columns=mconern.keys(), dtype=int).to_csv(f'confusion_mat{step}.csv', index=False)
                    # print(np.array(all_tags).ravel().shape, np.array(all_predicted_tags).ravel().shape)
        # print(all_tags, all_predicted_tags)
    writer.close()

