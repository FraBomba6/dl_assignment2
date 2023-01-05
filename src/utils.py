import pandas as pd
import torch.utils.data
import torch.backends.mps
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


def apply_random_validation_split(train_dataset: torch.utils.data.TensorDataset, split=20):
    units = len(train_dataset)
    validation_size = int(split * units / 100)
    train_size = units - validation_size
    assert(validation_size + train_size == units)

    return torch.utils.data.random_split(train_dataset, [train_size, validation_size])


def tokenize_for_multiple_choice(sentence_list, options_list, target_list):
    encode_plus_args = {
        "add_special_tokens": True,
        "max_length": 64,
        "padding": 'max_length',
        "return_attention_mask": True,
        "return_tensors": 'pt'
    }
    input_ids_list = []
    attention_masks_list = []
    return_target_list = []
    for (sentence, options, target_label) in tqdm(zip(sentence_list, options_list, target_list), total=len(sentence_list)):
        sentences = [(sentence, option) for option in options]
        encoded_dict = TOKENIZER(sentences, **encode_plus_args)
        input_ids_list.append(encoded_dict['input_ids'])
        attention_masks_list.append(encoded_dict['attention_mask'])
        return_target_list.append(int(target_label)-1)

    return torch.stack(input_ids_list, dim=0), torch.stack(attention_masks_list, dim=0), torch.as_tensor(return_target_list)
