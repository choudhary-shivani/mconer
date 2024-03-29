import numpy as np
import torch
from copy import deepcopy
from torch.nn.utils.rnn import pad_sequence
from torch import nn
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from utils.reader_utils import _assign_ner_tags, get_ner_reader, extract_spans
from lstmEncoder.customlstm import CustomLSTM
from lstmEncoder.Inferserial import InferSerialData
from trie.trie_build import build_trees, format_query_by_features

# all_tree = build_trees()
class CoNLLReader(Dataset):
    def __init__(self, max_instances: object = -1, max_length: object = 50, target_vocab: object = None,
                 pretrained_dir: object = '',
                 encoder_model: object = 'xlm-roberta-large', finegrained=True, reversemap=None) -> object:
        if reversemap is None:
            reversemap = {}
        self._max_instances = max_instances
        self._max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_dir + encoder_model)

        self.pad_token = self.tokenizer.special_tokens_map['pad_token']
        self.pad_token_id = self.tokenizer.get_vocab()[self.pad_token]
        self.sep_token = self.tokenizer.special_tokens_map['sep_token']

        self.label_to_id = {} if target_vocab is None else target_vocab
        self.instances = []
        self.fine = finegrained
        self.reversemap = reversemap
        self.lstmdata = InferSerialData()
        print("Loding LSTM model")
        # self.lstm = CustomLSTM(58, 128, num_layers=2).eval().to('cuda')
        # self.lstm.load_state_dict(torch.load(r'.\lstmEncoder\t_lstm_model.pt'))
        # self.all_tree = build_trees()
        print(self.fine)

    def get_target_size(self):
        return len(set(self.label_to_id.values()))

    def get_target_vocab(self):
        return self.label_to_id

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, item):
        return self.instances[item]

    def read_data(self, data):
        all_tags = []
        dataset_name = data if isinstance(data, str) else 'dataframe'
        print('Reading file {}'.format(dataset_name))
        instance_idx = 0

        for fields, metadata in get_ner_reader(data=data):
            all_tags += fields[-1]
            if self._max_instances != -1 and instance_idx > self._max_instances:
                break
            sentence_str, tokens_sub_rep, token_masks_rep, coded_ner_, gold_spans_, mask, lstm_encoded = self.parse_line_for_ner(
                fields=fields)
            print(sentence_str, gold_spans_)
            tokens_tensor = torch.tensor(tokens_sub_rep, dtype=torch.long)
            tag_tensor = torch.tensor(coded_ner_, dtype=torch.long).unsqueeze(0)
            token_masks_rep = torch.tensor(token_masks_rep)
            mask_rep = torch.tensor(mask)
            # print(tag_tensor)
            self.instances.append((tokens_tensor, mask_rep, token_masks_rep, gold_spans_, tag_tensor, lstm_encoded))
            instance_idx += 1
        print('Finished reading {:d} instances from file {}'.format(len(self.instances), dataset_name))
        # all_tags = list(np.unique(np.array(all_tags)))
        # print(sorted(all_tags, key=lambda x: x.split('-')[-1]))

    def parse_line_for_ner(self, fields):
        tokens_, ner_tags = fields[0], fields[-1]
        # print(tokens_)
        sentence_str, tokens_sub_rep, ner_tags_rep, token_masks_rep, mask, lstm_encoded = self.parse_tokens_for_ner(tokens_, ner_tags)
        gold_spans_ = extract_spans(ner_tags_rep)
        # print(ner_tags_rep)
        if not self.fine:
            coded_ner_ = [self.label_to_id[self.reversemap[tag]] if tag in self.reversemap else self.label_to_id['O']
                          for tag in ner_tags_rep]
            for key, val in gold_spans_.items():
                if f"B-{val}" in self.reversemap:
                    gold_spans_[key] = self.reversemap[f"B-{val}"].split('-')[-1]
                else:
                    # print(f"B-{val}")
                    gold_spans_[key] = 'O'
            # gold_spans_ = {k: self.reversemap[f"B-{v}"].split('-')[-1] for k, v in gold_spans_.items() }
        else:
            coded_ner_ = [self.label_to_id[tag] if tag in self.label_to_id else self.label_to_id['O'] for tag in
                          ner_tags_rep]

        return sentence_str, tokens_sub_rep, token_masks_rep, coded_ner_, gold_spans_, mask, lstm_encoded

    def parse_tokens_for_ner(self, tokens_, ner_tags):
        sentence_str = ''
        tokens_sub_rep, ner_tags_rep = [self.pad_token_id], ['O']
        token_masks_rep = [False]
        build_feature_rep = [[0] * (len(self.label_to_id)-1) ]
        for idx, token in enumerate(tokens_):
            if self._max_length != -1 and len(tokens_sub_rep) > self._max_length:
                break
            kb_feature = format_query_by_features(token, len(self.label_to_id)-1,
                                                  self.label_to_id, all_tree, 1)
            build_feature_rep.append(list(kb_feature[0]))

            sentence_str += ' ' + ' '.join(self.tokenizer.tokenize(token.lower()))
            rep_ = self.tokenizer(token.lower())['input_ids']
            rep_ = rep_[1:-1]
            tokens_sub_rep.extend(rep_)

            if len(rep_) > 1:
                # rotate the array as the labels are adjacent
                # print(list(kb_feature[0]))
                rot_feat = np.roll(kb_feature[0], 1)
                # [((kb_feature[0][-1] + kb_feature[0][:-1]]) * (len(rep_) - 1)
                build_feature_rep += ([list(rot_feat)] * (len(rep_) -1))
                # build_feature_rep += rot_feat
            # if we have a NER here, in the case of B, the first NER tag is the B tag, the rest are I tags.
            ner_tag = ner_tags[idx]
            tags, masks = _assign_ner_tags(ner_tag, rep_)

            ner_tags_rep.extend(tags)
            token_masks_rep.extend(masks)
            # print(token, rep_)
            # print(len(tokens_sub_rep))
            # print(len(build_feature_rep))
        tokens_sub_rep.append(self.pad_token_id)
        ner_tags_rep.append('O')
        token_masks_rep.append(False)
        mask = [True] * len(tokens_sub_rep)
        sent = ''
        # lstm_encoded.append(list(self.lstm.named_children())[0][1](torch.tensor(57).to('cuda')))
        # print(lstm_encoded)
        build_feature_rep.append(
            ([0] * (len(self.label_to_id)-1))
                                 )
        # print(sentence_str)
        # print(build_feature_rep.__len__(), len(tokens_sub_rep))
        # print(build_feature_rep)
        # all_sent_piece = []
        # for i in sentence_str.split():
        #     sent += i
        #     all_sent_piece.append(deepcopy(sent))
        # all_converted = self.lstmdata.convert(all_sent_piece)
        # all_converted = [torch.tensor([])] + all_converted
        # all_converted.append(torch.tensor([]))
        # encoded = pad_sequence(all_converted, padding_value=57, batch_first=True).int()
        # print(len(tokens_sub_rep))
        # print("encoded", encoded)

        # x, y = self.lstm(torch.tensor(encoded))
        # print(x.shape, .shape)
        # lstm_encoded = torch.mean(y, dim=0, keepdim=True).squeeze(0)
        # print(lstm_encoded.shape)
        # print(torch.softmax(lstm_encoded, dim=-1))
        # lstm_encoded.append(list(self.lstm.named_children())[0][1](torch.tensor(57).to('cuda')))
        # lstm_encoded = torch.stack(lstm_encoded, dim=2).squeeze(0)
        # print(lstm_encoded.shape)
        # x = torch.tensor([1,2,3])
        return sentence_str, tokens_sub_rep, ner_tags_rep, token_masks_rep, mask, build_feature_rep


if __name__ == '__main__':
    # wnut_iob = {'B-CORP': 0, 'I-CORP': 1, 'B-CW': 2, 'I-CW': 3, 'B-GRP': 4, 'I-GRP': 5, 'B-LOC': 6, 'I-LOC': 7,
    # 'B-PER': 8, 'I-PER': 9, 'B-PROD': 10, 'I-PROD': 11, 'O': 12} mconern = {'B-AerospaceManufacturer': 0,
    # 'I-AerospaceManufacturer': 1, 'B-AnatomicalStructure': 2, 'I-AnatomicalStructure': 3, 'B-ArtWork': 4,
    # 'I-ArtWork': 5, 'B-Artist': 6, 'I-Artist': 7, 'B-Athlete': 8, 'I-Athlete': 9, 'B-CarManufacturer': 10,
    # 'I-CarManufacturer': 11, 'B-Cleric': 12, 'I-Cleric': 13, 'B-Clothing': 14, 'I-Clothing': 15, 'B-Disease': 16,
    # 'I-Disease': 17, 'B-Drink': 18, 'I-Drink': 19, 'B-Facility': 20, 'I-Facility': 21, 'B-Food': 22, 'I-Food': 23,
    # 'B-HumanSettlement': 24, 'I-HumanSettlement': 25, 'B-MedicalProcedure': 26, 'I-MedicalProcedure': 27,
    # 'B-Medication/Vaccine': 28, 'I-Medication/Vaccine': 29, 'B-MusicalGRP': 30, 'I-MusicalGRP': 31,
    # 'B-MusicalWork': 32, 'I-MusicalWork': 33, 'O': 34, 'B-ORG': 35, 'I-ORG': 36, 'B-OtherLOC': 37, 'I-OtherLOC':
    # 38, 'B-OtherPER': 39, 'I-OtherPER': 40, 'B-OtherPROD': 41, 'I-OtherPROD': 42, 'B-Politician': 43,
    # 'I-Politician': 44, 'B-PrivateCorp': 45, 'I-PrivateCorp': 46, 'B-PublicCorp': 47, 'I-PublicCorp': 48,
    # 'B-Scientist': 49, 'I-Scientist': 50, 'B-Software': 51, 'I-Software': 52, 'B-SportsGRP': 53, 'I-SportsGRP': 54,
    # 'B-SportsManager': 55, 'I-SportsManager': 56, 'B-Station': 57, 'I-Station': 58, 'B-Symptom': 59, 'I-Symptom':
    # 60, 'B-Vehicle': 61, 'I-Vehicle': 62, 'B-VisualWork': 63, 'I-VisualWork': 64, 'B-WrittenWork': 65,
    # 'I-WrittenWork': 66}
    from tutils import invert, indvidual
    from tutils import mconer_grouped

    fine = True
    mconern = indvidual(mconer_grouped, fine)
    reveremap = invert(mconer_grouped)
    print(mconern, reveremap)
    ds = CoNLLReader(target_vocab=mconern, finegrained=fine, reversemap=reveremap)
    # # ds.read_data(data=r'C:\Users\Rah12937\PycharmProjects\mconer\multiconer2023\train_dev\en-train.conll')
    ds.read_data(data=r'C:\Users\Rah12937\PycharmProjects\mconer\multiconer2023\train_dev\en_dev_small.conll')
    for i in range(len(ds.instances)):
        tokens_tensor, mask_rep, token_masks_rep, gold_spans_, tag_tensor, lstm_encoded = ds.instances[i]
        # print(lstm_encoded.shape)
        # print(torch.argmax(torch.softmax(lstm_encoded, dim=-1), dim=-1), gold_spans_)
        print(torch.tensor(lstm_encoded), len(tokens_tensor))
