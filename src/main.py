from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import utils
import os
from rich.console import Console

console = Console()
# %%
console.log("Loading data")
train_file = os.path.join(utils.PROJECT_ROOT, "data", "train.jsonl")
test_file = os.path.join(utils.PROJECT_ROOT, "data", "test.jsonl")

train_data = utils.load_train(train_file)
test_data = utils.load_train(test_file)

# %%
console.log("Tokenizing train data")
train_input_ids, train_attention_masks, train_targets = utils.tokenize(
    train_data["sentence"].to_list(),
    train_data[["option1", "option2"]].values.tolist(),
    train_data["answer"].to_list()
)

console.log("Tokenizing test data")
test_input_ids, test_attention_masks, test_targets = utils.tokenize(
    test_data["sentence"].to_list(),
    test_data[["option1", "option2"]].values.tolist(),
    test_data["answer"].to_list()
)

# %%
console.log("Creating datasets and dataloaders")
train_dataset, validation_dataset = utils.apply_random_validation_split(TensorDataset(train_input_ids, train_attention_masks, train_targets))
train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=4)
validation_dataloader = DataLoader(validation_dataset, sampler=RandomSampler(validation_dataset), batch_size=4)

test_dataset = TensorDataset(test_input_ids, test_attention_masks, test_targets)
test_dataloader = DataLoader(test_dataset, sampler=RandomSampler(test_dataset), batch_size=4)

# %%

