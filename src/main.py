import utils
import os

# %%
train_file = os.path.join(utils.PROJECT_ROOT, "data", "train.jsonl")
test_file = os.path.join(utils.PROJECT_ROOT, "data", "test.jsonl")

train_data = utils.load_train(train_file)

train, validation = utils.apply_random_validation_split(train_data)

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# %%
train_input_ids, train_attention_masks, train_targets = utils.tokenize(
    train_data["sentence"].to_list(),
    train_data["answer"].to_list()
)
