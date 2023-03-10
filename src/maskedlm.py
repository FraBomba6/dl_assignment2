import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from transformers import BertForMaskedLM
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import numpy as np
import random
import utils
import os
from rich.console import Console

console = Console()

# %%
BATCH_SIZE = 32
EPOCHS = 2

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# %%
console.log("Loading data")
train_file = os.path.join(utils.PROJECT_ROOT, "data", "train.jsonl")
test_file = os.path.join(utils.PROJECT_ROOT, "data", "test.jsonl")

train_data = utils.load_data(train_file)
test_data = utils.load_data(test_file)

# %%
console.log("Tokenizing train data")
train_input_ids, train_attention_masks, train_targets, train_opt_index_list, train_options = utils.tokenize_for_mlm(
    train_data["sentence"].to_list(),
    train_data[["option1", "option2"]].values.tolist(),
    train_data["answer"].to_list()
)

console.log("Tokenizing test data")
test_input_ids, test_attention_masks, test_targets, test_opt_index_list, test_options = utils.tokenize_for_mlm(
    test_data["sentence"].to_list(),
    test_data[["option1", "option2"]].values.tolist(),
    test_data["answer"].to_list()
)

# %%
console.log("Creating datasets and dataloaders")
train_dataset, validation_dataset = utils.apply_random_validation_split(
    TensorDataset(
        train_input_ids,
        train_attention_masks,
        train_targets,
        train_opt_index_list,
        train_options
    )
)
train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=BATCH_SIZE)
validation_dataloader = DataLoader(validation_dataset, sampler=RandomSampler(validation_dataset), batch_size=BATCH_SIZE)

test_dataset = TensorDataset(test_input_ids, test_attention_masks, test_targets, test_opt_index_list, test_options)
test_dataloader = DataLoader(test_dataset, sampler=RandomSampler(test_dataset), batch_size=BATCH_SIZE)

# %%
console.log("Clearing variables")
del(train_input_ids, train_attention_masks, train_targets, train_opt_index_list, train_options)
del(test_input_ids, test_attention_masks, test_targets, test_opt_index_list, test_options)

# %%
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.to(utils.DEVICE)

optimizer = AdamW(model.parameters(), lr=5e-3, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=EPOCHS * len(train_dataloader)
)


# %%
def mlm_accuracy(preds, labels):
    labels = np.array(labels.tolist())
    return np.sum(np.array(preds) == labels) / BATCH_SIZE


def filter_options(batched_input: torch.Tensor, batched_options: torch.Tensor, batched_logits: torch.Tensor):
    answers = []
    for j in range(BATCH_SIZE):
        mask_index = (batched_input[i] == 103).nonzero(as_tuple=True)[0]
        option1_index = batched_options[i][0][0]
        option2_index = batched_options[i][1][0]
        option1_prob = batched_logits[i][mask_index][0][option1_index]
        option2_prob = batched_logits[i][mask_index][0][option2_index]
        answers.append(0 if option1_prob > option2_prob else 1)
    return answers


def train_bert_one_epoch(dataloader, epoch):
    console.log(f"Training epoch #{epoch+1}")
    total_loss = 0
    model.train()

    for step, batch in utils.tqdm(enumerate(dataloader), total=len(dataloader)):
        batch_input_ids = batch[0].to(utils.DEVICE)
        batch_input_masks = batch[1].to(utils.DEVICE)
        batch_labels = batch[2].to(utils.DEVICE)

        model.zero_grad()
        outputs = model(
            input_ids=batch_input_ids,
            attention_mask=batch_input_masks,
            labels=batch_labels
        )
        loss, logits = outputs[:2]
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    avg_loss = total_loss / len(dataloader)
    console.log("Average loss: {0:.4f}".format(avg_loss))


def test_bert(dataloader):
    console.log(f"Testing")
    total_loss = 0
    total_accuracy = 0
    model.eval()

    for step, batch in utils.tqdm(enumerate(dataloader), total=len(dataloader)):
        batch_input_ids = batch[0].to(utils.DEVICE)
        batch_input_masks = batch[1].to(utils.DEVICE)
        batch_labels = batch[2].to(utils.DEVICE)
        batch_labels_index = batch[3]
        batch_options = batch[4]

        with torch.no_grad():
            outputs = model(
                input_ids=batch_input_ids,
                attention_mask=batch_input_masks,
                labels=batch_labels
            )
            loss, logits = outputs[:2]
            total_loss += loss.item()

        model_answers = filter_options(batch_input_ids, batch_options, logits)
        total_accuracy += mlm_accuracy(model_answers, batch_labels_index)

    avg_loss = total_loss / len(dataloader)
    console.log("Average loss: {0:.4f}".format(avg_loss))
    avg_accuracy = total_accuracy / len(dataloader)
    console.log("Accuracy: {0:.4f}".format(avg_accuracy))


# %%
for i in range(EPOCHS):
    train_bert_one_epoch(train_dataloader, i)
    test_bert(train_dataloader)
    test_bert(validation_dataloader)
    test_bert(test_dataloader)
