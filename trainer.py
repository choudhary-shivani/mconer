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
from pytorchtools import EarlyStopping
from torch.optim.lr_scheduler import ReduceLROnPlateau
# from torch.utils.tensorboard import SummaryWriter
from utils import *
from tensorboardX import SummaryWriter
# from utils import mconern, encoder_model, tokenizer
from reader import *
# from tutils import image_gen
from NERmodel3 import NERmodelbase3
from tqdm import tqdm
from kornia.losses import FocalLoss
from torchmetrics.classification import MulticlassConfusionMatrix, MulticlassRecall, MulticlassPrecision, \
    MulticlassF1Score

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
    lstm_encoded_tensor = torch.zeros(size=(len(tokens), max_len, 72), dtype=torch.float)
    # lstm_encoded = torch.tensor(lstm_encoded)
    # print(lstm_encoded.shape)
    for i in range(len(tokens)):
        temp = torch.tensor(lstm_encoded[i])
        tokens_ = tokens[i]
        seq_len = len(tokens_)

        token_tensor[i, :seq_len] = tokens_
        tag_tensor[i, :seq_len] = tags[i]
        mask_tensor[i, :seq_len] = masks[i]
        token_masks_tensor[i, :seq_len] = token_masks[i]
        # lstm_encoded_tensor[i, 1:seq_len - 1, :] = lstm_encoded[i]
        # modifying so that each word has separate token
        for j in range(temp.__len__()):
            lstm_encoded_tensor[i, j, :] = temp[j]
    # print(lstm_encoded_tensor)
    return token_tensor, tag_tensor, mask_tensor, token_masks_tensor, gold_spans, lstm_encoded_tensor


def trainer(NUM_EPOCH, BATCH_SIZE, fine, force, eval_step=100, ratio=0.2):
    print(fine)
    if force:
        from dataloader import CoNLLReader
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

    validloader = DataLoader(valid, batch_size=BATCH_SIZE, collate_fn=collate_batch, num_workers=0)

    # next(iter(validloader))
    # sys.exit(2)
    trainloader = DataLoader(ds, batch_size=BATCH_SIZE, collate_fn=collate_batch, num_workers=0, shuffle=True)
    model = NERmodelbase3(tag_to_id=mconern, device=device, encoder_model=encoder_model, dropout=0.3, use_lstm=True).to(
        device)
    print(model)
    WARMUP_STEP = int(len(trainloader) * NUM_EPOCH * 0.1) // 100
    print(f"Number of warm up step is {WARMUP_STEP}")
    optim, scheduler = get_optimizer(model, True, warmup=WARMUP_STEP)
    scheduler2 = ReduceLROnPlateau(optimizer=optim[0], factor=0.01, patience=3, verbose=True, cooldown=1)

    run_id = random.randint(1, 10000)
    run_name = f"runid_{run_id}_EP_{NUM_EPOCH}_fine_xlm-b-birnnn-fl-{0.8}-sep_lr-alpha-2-gama-4"
    while True:
        if os.path.exists(run_name):
            run_id = random.randint(1, 10000)
            run_name = f"runid_{run_id}_EP_{NUM_EPOCH}_fine_xlm-b-birnnn-fl-{0.8}-sep_lr-alpha-2-gama-4-tf"
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
                outputs, focal_loss, all_prob, token_scores, mask, tags = model(data)
                loss = 0.4 * outputs['loss'] + 0.6 * focal_loss
                       # - 0.2 * outputs['results']['micro@F1']
                running_loss += loss
                loss.backward()
                optim[0].step()
                if (step + 1) % 50 == 0:
                    scheduler[0].step()
                model.spanf1.reset()
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
                                outputs, focal_loss, all_prob, token_scores, mask, tags = model(data, mode='predict')
                                # print(outputs['loss'].detach(), focal_loss, focal_loss_v,
                                #       (0.2 * outputs['loss'].detach() + 0.4 * focal_loss + 0.4 * focal_loss_v))
                                val_loss += 0.4 * outputs['loss'] + 0.6 * focal_loss
                                            # - 0.2 * outputs['results']['micro@F1']
                                # val_loss += (0.2 * outputs['loss'].detach() + 0.4 * focal_loss + 0.4 * focal_loss_v)
                                # val_loss = closs
                    print("\n", outputs['results']['micro@F1'])
                    # print("\n", val_loss)
                    model.train()
                    writer.add_scalars("Loss",
                                       {
                                           "Train Loss": round(running_loss.detach().cpu().numpy().ravel()[0] / 20, 4),
                                           "Valid Loss": round(val_loss.detach().cpu().numpy().ravel()[0] / 20, 4),
                                       }
                                       , step)
                    # print(outputs['results'])
                    writer.add_scalars("Metrics", outputs['results'], step)
                    val_track.append(round(val_loss.detach().cpu().numpy().ravel()[0] / len(validloader), 4))
                    running_loss = 0
                    # if round(outputs['results']['micro@F1'], 6) > 1e-2:
                    #     early_stopping(-1* round(outputs['results']['micro@F1'], 6), model)
                    early_stopping(round(val_loss.detach().cpu().numpy().ravel()[0] / len(validloader), 6), model)
                    scheduler2.step(round(val_loss.detach().cpu().numpy().ravel()[0] / len(validloader), 6))
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
