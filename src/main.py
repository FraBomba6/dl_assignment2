from transformers import BertTokenizer
import utils
import os

train_file = os.path.join(utils.PROJECT_ROOT, "data", "train.jsonl")
train_file = os.path.join(utils.PROJECT_ROOT, "data", "test.jsonl")

train_data = utils.load_train(train_file)

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
