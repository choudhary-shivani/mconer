import torch
from trainer import *
from dataloader import CoNLLReader, get_ner_reader, extract_spans
from utils.metric import SpanF1

force = False
fine = True
device = 'cuda'

if os.path.exists('valid_load.pkl') and not force:
    with open('valid_load.pkl', 'rb') as f:
        valid = pickle.load(f)
else:
    valid = CoNLLReader(target_vocab=mconern, encoder_model=encoder_model, reversemap=reveremap, finegrained=fine)
    valid.read_data(data=r'C:\Users\Rah12937\PycharmProjects\mconer\multiconer2023\train_dev\en-dev.conll')
    with open('valid_load.pkl', 'wb') as f:
        pickle.dump(valid, f)
def calculate_macro_f1(label_file, pred_file):
    labels = [field for field in get_ner_reader(label_file)]
    preds = [field for field in get_ner_reader(pred_file)]

    # pdb.set_trace()

    pred_result = []
    label_result = []
    for pred, label in zip(preds, labels):
        print(pred_result)
        label_result.append(extract_spans(label[-1]))
        pred_result.append(extract_spans(pred[-1]))

    span_f1 = SpanF1()
    span_f1(pred_result, label_result)
    word_result = span_f1.get_metric()

    return word_result["macro@F1"]

file_path = r'C:\Users\Rah12937\PycharmProjects\mconer\multiconer2023\train_dev\en-dev.conll'
model = NERmodelbase3(tag_to_id=mconern, device=device, encoder_model=encoder_model, dropout=0.3, use_lstm=True).to(
        device)
model.load_state_dict(torch.load(r'C:\Users\Rah12937\PycharmProjects\mconer'
                                         r'\runid_8303_EP_20_fine_xlm-b-birnnn-fl-0.8-sep_lr-alpha-2-gama-4'
                                         r'\runid_8303_EP_20_fine_xlm-b-birnnn-fl-0.8-sep_lr-alpha-2-gama-4.pt'))

validloader = DataLoader(valid, batch_size=32, collate_fn=collate_batch, num_workers=0)
out_str = ''

eval_file = os.path.join('test_out.txt')
for batch in tqdm(validloader, total=1):
    outputs, focal_loss, all_prob, token_scores, mask, tags = model(batch, mode='predict')
    # convert to the required format for the user
    for idx, record in enumerate(batch[0]):
        # convert ids to token so that we can undertsnd the toke's behav
        words = tokenizer.convert_ids_to_tokens(record)
        preds = outputs['token_tags'][idx]
        slicer = [True if i.startswith('▁') else False for i in words[:len(preds)] ]
        filter_preds = np.array(preds)[np.array(slicer)]
        out_str += '\n'.join(filter_preds)
        out_str += '\n\n'
open(eval_file, 'wt').write(out_str)

# macro_f1_score = calculate_macro_f1(file_path, eval_file)
# score_file = os.path.join('score_out.txt')
# open(score_file, 'wt').write(str(macro_f1_score))