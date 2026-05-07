"""
PyTorch Dataset for the tweet classification task.

Each sample consists of:
  - tokenized tweet text (BERT input_ids, attention_mask, token_type_ids)
  - label  (0 = down, 1 = flat, 2 = up)
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import BertTokenizer

from config import BERT_MODEL, MAX_LENGTH


class TweetDataset(Dataset):
    """
    Map-style dataset wrapping a DataFrame of labeled tweets.

    Parameters
    ----------
    df        : DataFrame with at minimum columns 'text' and 'label'.
    tokenizer : Pre-loaded BertTokenizer; loaded from BERT_MODEL if None.
    """

    def __init__(self, df: pd.DataFrame, tokenizer: BertTokenizer = None):
        self.df        = df.reset_index(drop=True)
        self.tokenizer = tokenizer or BertTokenizer.from_pretrained(BERT_MODEL)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row   = self.df.iloc[idx]
        text  = str(row["text"])
        label = int(row["label"])          # 0 = down | 1 = flat | 2 = up

        enc = self.tokenizer(
            text,
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids":      enc["input_ids"].squeeze(0),           # (MAX_LENGTH,)
            "attention_mask": enc["attention_mask"].squeeze(0),      # (MAX_LENGTH,)
            "token_type_ids": enc["token_type_ids"].squeeze(0),      # (MAX_LENGTH,)
            "label":          torch.tensor(label, dtype=torch.long),
        }
