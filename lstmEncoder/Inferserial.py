import unicodedata
import torch
from torch.utils.data import Dataset
import string


class InferSerialData(Dataset):
    def __init__(self):
        self.all_chars = string.ascii_letters + " .,;'"
        self.num_char = len(self.all_chars)
        self.all_chars_dict = {v: k for k, v in enumerate(self.all_chars)}
        self.all_instances = []

    def unicodeToAscii(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
            and c in self.all_chars
        )

    def convert(self, line):
        temp = map(lambda a: self.all_chars_dict.get(a, self.num_char), self.unicodeToAscii(line))
        all_data = torch.tensor(list(temp))
        if all_data.size()[0] == 0:
            all_data = torch.tensor([57])
        return all_data
