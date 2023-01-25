import torch
import string
import glob
import unicodedata
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class Serialdata(Dataset):
    def __init__(self, filedir):
        self.all_chars = string.ascii_letters + " .,;'"
        self.num_char = len(self.all_chars)
        self.all_data = pd.read_csv(filedir, sep='\t').dropna()
        self.classes = self.all_data['label'].unique()
        self.num_classes = len(self.classes)
        self.all_data = self.all_data[:10000]
        self.tag_to_id = {v: k for k, v in enumerate(self.classes)}
        self.all_data['mapped'] = self.all_data['combined'].apply(self.unicodeToAscii)
        self.all_chars_dict = {v: k for k, v in enumerate(self.all_chars)}
        self.train, self.test = train_test_split(self.all_data, test_size=0.3, shuffle=True, random_state=42)
        self.all_instances = []
        print(self.tag_to_id)
    def unicodeToAscii(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
            and c in self.all_chars
        )

    def readLines(self, train):
        if train:
            for i, j in zip(self.train['mapped'].to_list(), self.train['label'].to_list()):
                temp = map(lambda a: self.all_chars_dict.get(a, self.num_char), i)
                self.all_instances.append((torch.tensor(list(temp)), self.tag_to_id[j]))
        else:
            for i, j in zip(self.test['mapped'].to_list(), self.test['label'].to_list()):
                temp = map(lambda a: self.all_chars_dict.get(a, self.num_char), i)
                self.all_instances.append((torch.tensor(list(temp)), self.tag_to_id[j]))
        # for i in self.all_instances:
        #     print(i)

    def readLines_infer(self, line):
        temp = map(lambda a: self.all_chars_dict.get(a, self.num_char), i)
        return torch.tensor(list(temp))

    def __len__(self):
        return len(self.all_instances)

    def __getitem__(self, item):
        return self.all_instances[item]


if __name__ == "__main__":
    sd = Serialdata('combined.tsv')
    sd.readLines()
    # print(sd[1])