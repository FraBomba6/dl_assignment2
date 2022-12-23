import pandas as pd
import torch
import json
import os
from transformers import BertTokenizer
from tqdm import tqdm

# Defining project root in order to avoid relative paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Initializing torch device according to hardware available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)


def load_train(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Unable to find {path}")
    with open(path, "r") as f:
        output = [json.loads(line) for line in f.readlines()]
        return pd.DataFrame.from_records(output)


def tokenize(sentence_list, target_list):
    encode_plus_args = {
        "add_special_tokens": True,
        "max_length": 45,
        "padding": 'max_length',
        "return_attention_mask": True,
        "return_tensors": 'pt',
        "truncation": True,
    }
    input_ids_list = []
    attention_masks_list = []
    return_target_list = []
    for (sentence, target_label) in tqdm(zip(sentence_list, target_list), total=len(sentence_list)):
        encoded_dict = TOKENIZER.encode_plus(sentence, **encode_plus_args)
        input_ids_list.append(encoded_dict['input_ids'])
        attention_masks_list.append(encoded_dict['attention_mask'])
        return_target_list.append(int(target_label))

    return torch.cat(input_ids_list, dim=0), torch.cat(attention_masks_list, dim=0), torch.as_tensor(return_target_list)
