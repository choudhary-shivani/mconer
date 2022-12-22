import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from utils.reader_utils import _assign_ner_tags, get_ner_reader, extract_spans


class CoNLLReader(Dataset):
    def __init__(self, max_instances: object = -1, max_length: object = 50, target_vocab: object = None, pretrained_dir: object = '',
                 encoder_model: object = 'xlm-roberta-large') -> object:
        self._max_instances = max_instances
        self._max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_dir + encoder_model)

        self.pad_token = self.tokenizer.special_tokens_map['pad_token']
        self.pad_token_id = self.tokenizer.get_vocab()[self.pad_token]
        self.sep_token = self.tokenizer.special_tokens_map['sep_token']

        self.label_to_id = {} if target_vocab is None else target_vocab
        self.instances = []

    def get_target_size(self):
        return len(set(self.label_to_id.values()))

    def get_target_vocab(self):
        return self.label_to_id

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, item):
        return self.instances[item]

    def read_data(self, data):
        dataset_name = data if isinstance(data, str) else 'dataframe'
        print('Reading file {}'.format(dataset_name))
        instance_idx = 0

        for fields, metadata in get_ner_reader(data=data):
            if self._max_instances != -1 and instance_idx > self._max_instances:
                break
            sentence_str, tokens_sub_rep, token_masks_rep, coded_ner_, gold_spans_, mask = self.parse_line_for_ner(fields=fields)

            tokens_tensor = torch.tensor(tokens_sub_rep, dtype=torch.long)
            tag_tensor = torch.tensor(coded_ner_, dtype=torch.long).unsqueeze(0)
            token_masks_rep = torch.tensor(token_masks_rep)
            mask_rep = torch.tensor(mask)

            self.instances.append((tokens_tensor, mask_rep, token_masks_rep, gold_spans_, tag_tensor))
            instance_idx += 1
        print('Finished reading {:d} instances from file {}'.format(len(self.instances), dataset_name))

    def parse_line_for_ner(self, fields):
        tokens_, ner_tags = fields[0], fields[-1]
        sentence_str, tokens_sub_rep, ner_tags_rep, token_masks_rep, mask = self.parse_tokens_for_ner(tokens_, ner_tags)
        gold_spans_ = extract_spans(ner_tags_rep)
        coded_ner_ = [self.label_to_id[tag] if tag in self.label_to_id else self.label_to_id['O'] for tag in ner_tags_rep]

        return sentence_str, tokens_sub_rep, token_masks_rep, coded_ner_, gold_spans_, mask

    def parse_tokens_for_ner(self, tokens_, ner_tags):
        sentence_str = ''
        tokens_sub_rep, ner_tags_rep = [self.pad_token_id], ['O']
        token_masks_rep = [False]
        for idx, token in enumerate(tokens_):
            if self._max_length != -1 and len(tokens_sub_rep) > self._max_length:
                break
            sentence_str += ' ' + ' '.join(self.tokenizer.tokenize(token.lower()))
            rep_ = self.tokenizer(token.lower())['input_ids']
            rep_ = rep_[1:-1]
            tokens_sub_rep.extend(rep_)

            # if we have a NER here, in the case of B, the first NER tag is the B tag, the rest are I tags.
            ner_tag = ner_tags[idx]
            tags, masks = _assign_ner_tags(ner_tag, rep_)

            ner_tags_rep.extend(tags)
            token_masks_rep.extend(masks)

        tokens_sub_rep.append(self.pad_token_id)
        ner_tags_rep.append('O')
        token_masks_rep.append(False)
        mask = [True] * len(tokens_sub_rep)
        return sentence_str, tokens_sub_rep, ner_tags_rep, token_masks_rep, mask


if __name__ == '__main__':
    wnut_iob = {'B-CORP': 0, 'I-CORP': 1, 'B-CW': 2, 'I-CW': 3, 'B-GRP': 4, 'I-GRP': 5, 'B-LOC': 6, 'I-LOC': 7,
                'B-PER': 8, 'I-PER': 9, 'B-PROD': 10, 'I-PROD': 11, 'O': 12}
    ds = CoNLLReader(target_vocab=wnut_iob)
    ds.read_data(data=r'C:\Users\Rah12937\PycharmProjects\mconern\multiconer2022\EN-English\en_dev.conll')