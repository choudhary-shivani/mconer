import numpy as np
import sys
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
from tqdm import tqdm

# sys.exit(0)
run_id = random.randint(1, 10000)
writer = SummaryWriter(f"runid_{run_id}/_EP_{NUM_EPOCH}__num_bs_{BATCH_SIZE}")
step = 0
running_loss = 0
early_stopping = EarlyStopping(patience=6, verbose=True, path='fine_xlm-b.pt')
eval_step = 50
# print(next(iter(trainloader)))

print(model)
# sys.exit(2)
for epoch in range(NUM_EPOCH):
    val_track = []
    with tqdm(trainloader, unit='batch') as tepoch:
        # model.train()
        tepoch.set_description(f"Epoch {epoch}")
        for i, data in enumerate(tepoch):
            optim[0].zero_grad()
            outputs = model(data)
            loss = outputs['loss']
            running_loss += loss
            loss.backward()
            optim[0].step()
            scheduler[0].step()
            # if i % 10 == 0:  # print every 2000 mini-batches
            model.spanf1.reset()
            writer.add_scalar('lr', scheduler[0].get_last_lr()[0], step)
            # run validation
            step += 1
            if (step + 1) % eval_step == 0:
                model.eval()
                with torch.no_grad():
                    with tqdm(validloader, unit='batch') as tepoch:
                        val_loss = 0

                        for i, data in enumerate(tepoch):
                            outputs = model(data, mode='predict')
                            val_loss += outputs['loss']
                model.train()
                writer.add_scalars("Loss",
                                   {
                                       "Train Loss": round(running_loss.detach().cpu().numpy().ravel()[0] / 20, 4),
                                       "Valid Loss": round(val_loss.detach().cpu().numpy().ravel()[0] / 20, 4),
                                   }
                                   , step)
                writer.add_scalars("Metrics", outputs['results'], step)
                # print(outputs['results'])
                # writer.add_scalar("Loss/Test", round(val_loss.numpy()[0] / len(validloader), 4), step)
                # writer.add_scalar("Loss/Valid",
                #                   round(val_loss.detach().cpu().numpy().ravel()[0] / 20, 4), step)
                val_track.append(round(val_loss.detach().cpu().numpy().ravel()[0] / eval_step, 4))
                running_loss = 0

                early_stopping(round(val_loss.detach().cpu().numpy().ravel()[0] / eval_step, 4), model)
                if early_stopping.early_stop:
                    print("Stopping early")
                    break

writer.close()
