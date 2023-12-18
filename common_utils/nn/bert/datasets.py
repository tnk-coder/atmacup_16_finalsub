from tqdm import tqdm
import pandas as pd
from torch.utils.data import Dataset
import torch
from typing import List

from transformers import AutoTokenizer

def build_tokenizer(cfg, mode):
    if mode == 'train':
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_first_name + cfg.model_name)

        tokenizer.truncation_side = cfg.truncation_side
        print(tokenizer.truncation_side)

        tokenizer.save_pretrained(cfg.model_dir + 'tokenizer/')
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_dir + 'tokenizer/')

    return tokenizer

def get_inputs_list(df_token_cls: pd.DataFrame, tokenizer, max_length: int, target_col: str, text_cols: List[str], mode='train'):
    if len(text_cols) == 1:
        text = df_token_cls[text_cols[0]]
    else:
        text = df_token_cls[text_cols[0]].str.cat(
            df_token_cls[text_cols[1:]], sep=tokenizer.sep_token)

    text = text.values
    print(text)

    inputs_list = []
    for i in tqdm(range(len(text))):
        inputs = tokenizer.encode_plus(
            text[i],
            truncation=True,
            add_special_tokens=True,
            # padding="max_length",
            max_length=max_length  # CFG.max_length
        )
        inputs_list.append(inputs)

    len_list = [len(inputs_list[i]['input_ids'])
                for i in range(len(inputs_list))]

    if mode == 'train':
        for i in range(len(inputs_list)):
            inputs_list[i]['label'] = df_token_cls.iloc[i][target_col]

    return inputs_list, len_list

class CustomDataset(Dataset):
    def __init__(self, df, inputs, labels=None, ):
        self.df = df

        # tokenizer, sep_token = build_tokenizer(cfg, mode)
        # self.tokenizer = tokenizer
        # self.sep_token = sep_token

        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        inputs = self.inputs[idx]

        # collateあり
        if self.labels is None:
            return inputs

        label = self.labels[idx]

        inputs['label'] = label
        return inputs
