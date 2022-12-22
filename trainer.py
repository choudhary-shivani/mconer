import torch
from transformers import get_constant_schedule_with_warmup
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from NERmodel import NERmodelbase
from dataloader import CoNLLReader
encoder_model = 'xlm-roberta-large'
tokenizer = AutoTokenizer.from_pretrained('' + encoder_model)


def collate_batch(batch):
    batch_ = list(zip(*batch))
    tokens, masks, token_masks, gold_spans, tags = batch_[0], batch_[1], batch_[2], batch_[3], batch_[4]

    max_len = max([len(token) for token in tokens])
    token_tensor = torch.empty(size=(len(tokens), max_len), dtype=torch.long).fill_(tokenizer.pad_token_id)
    tag_tensor = torch.empty(size=(len(tokens), max_len), dtype=torch.long).fill_(wnut_iob['O'])
    mask_tensor = torch.zeros(size=(len(tokens), max_len), dtype=torch.bool)
    token_masks_tensor = torch.zeros(size=(len(tokens), max_len), dtype=torch.bool)

    for i in range(len(tokens)):
        tokens_ = tokens[i]
        seq_len = len(tokens_)

        token_tensor[i, :seq_len] = tokens_
        tag_tensor[i, :seq_len] = tags[i]
        mask_tensor[i, :seq_len] = masks[i]
        token_masks_tensor[i, :seq_len] = token_masks[i]

    return token_tensor, tag_tensor, mask_tensor, token_masks_tensor, gold_spans


def get_optimizer(net, opt=False):
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=0.03)
    if opt:
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=2)
        return [optimizer], [scheduler]
    return [optimizer]


NUM_EPOCH = 10
BATCH_SIZE = 64

wnut_iob = {'B-CORP': 0, 'I-CORP': 1, 'B-CW': 2, 'I-CW': 3, 'B-GRP': 4, 'I-GRP': 5, 'B-LOC': 6, 'I-LOC': 7,
            'B-PER': 8, 'I-PER': 9, 'B-PROD': 10, 'I-PROD': 11, 'O': 12}
ds = CoNLLReader(target_vocab=wnut_iob)
ds.read_data(data=r'C:\Users\Rah12937\PycharmProjects\mconern\multiconer2022\EN-English\en_train:q.conll')

valid = CoNLLReader(target_vocab=wnut_iob)
valid.read_data(data=r'C:\Users\Rah12937\PycharmProjects\mconern\multiconer2022\EN-English\en_dev.conll')

test = CoNLLReader(target_vocab=wnut_iob)
test.read_data(data=r'C:\Users\Rah12937\PycharmProjects\mconern\multiconer2022\EN-English\en_dev.conll')


device = 'cpu'
# if torch.cuda.is_available():
#     device = 'cuda'

model = NERmodelbase(tag_to_id=wnut_iob, device=device ).to(device)
criterion = torch.nn.CrossEntropyLoss()
optim, scheduler = get_optimizer(model, True)

trainloader = DataLoader(ds, batch_size=BATCH_SIZE, collate_fn=collate_batch, num_workers=0)
validloader = DataLoader(valid, batch_size=BATCH_SIZE, collate_fn=collate_batch, num_workers=0)

from tqdm import tqdm

for epoch in range(NUM_EPOCH):
    with tqdm(trainloader, unit='batch') as tepoch:
        tepoch.set_description(f"Epoch {epoch}")
        running_loss = 0
        for i, data in enumerate(tepoch):
            optim[0].zero_grad()
            outputs = model(data)
            loss = outputs['loss']
            running_loss += loss
            loss.backward()
            optim[0].step()
            if i % 10 == 0:  # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 5:.3f}')
                running_loss = 0.0
                print(f"{outputs['results']}")