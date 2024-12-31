import datasets
import transformers
import evaluate
import accelerate

from datasets import load_dataset, DatasetDict
from transformers import DataCollatorWithPadding
from transformers import AutoTokenizer
from pprint import pprint

irony_dataset = load_dataset("cardiffnlp/tweet_eval", "irony")

train_dataset = irony_dataset['train'][:50]
validation_dataset = irony_dataset['validation']
test_dataset = irony_dataset['test']

# Print the dataset structure to verify
# pprint(train_dataset)

#for item in train_dataset:
#  print(item["label"], item["text"])

#for i in range(len(train_dataset['text'])):
#  print(train_dataset['label'][i], train_dataset['text'][i])

for label, text in zip(train_dataset['label'], train_dataset['text']):
    print(label, text)

