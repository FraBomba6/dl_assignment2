import pandas as pd
import torch.utils.data
import torch.backends.mps
import torch
import json
import os

# Defining project root in order to avoid relative paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Initializing torch device according to hardware available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def load_train(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Unable to find {path}")
    with open(path, "r") as f:
        output = [json.loads(line) for line in f.readlines()]
        return pd.DataFrame.from_records(output)

def apply_random_validation_split(train_data, split=20):
    units = train_data.shape[0] 
    validation_size = split * units / 100
    train_size = units - validation_size
    assert(validation_size + train_size == units)

    return torch.utils.data.random_split(train_data, [train_size, validation_size])

