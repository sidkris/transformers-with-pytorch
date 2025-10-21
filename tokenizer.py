import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
text = Path("sherlock-holmes.txt").read_text(encoding = "utf-8")


class CharTokenizer:
    def __init__(self, vocabulary):
        self.token_id_for_char = {char : token_id for token_id, char in enumerate(vocabulary)}
        self.char_for_token_id = {token_id : char for token_id, char in enumerate(vocabulary)}

    @staticmethod
    def train_from_text(text):
        vocabulary = set(text)
        return CharTokenizer(sorted(list(vocabulary)))

    def encode(self, text):
        token_ids = []
        for char in text :
            token_ids.append(self.token_id_for_char[char])
        return torch.tensor(token_ids, dtype = torch.long)

    def decode(self, token_ids):
        chars = []
        for token_id in token_ids :
            chars.append(self.char_for_token_id[token_id])
        return "".join(chars)

    def vocabulary_size(self):
        return len(self.token_id_for_char)
    

if __name__ == "__main__" :
    tokenizer = CharTokenizer.train_from_text(text)
    print(tokenizer.encode("Hello World"))
    print(tokenizer.vocabulary_size())