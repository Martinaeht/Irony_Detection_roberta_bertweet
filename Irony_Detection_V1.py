'''Adapted from a COLAB Notebook'''

import datasets
import transformers
import evaluate
import accelerate

from datasets import load_dataset, DatasetDict
from transformers import DataCollatorWithPadding
from transformers import AutoTokenizer
from pprint import pprint
import numpy as np
import evaluate
from transformers import TrainingArguments, Trainer
from transformers import AutoModelForSequenceClassification
from transformers import set_seed
import torch

irony_dataset = load_dataset("cardiffnlp/tweet_eval", "irony")

train_dataset = irony_dataset['train']
validation_dataset = irony_dataset['validation']
test_dataset = irony_dataset['test']

# Print the dataset structure to verify
# pprint(train_dataset)

#for item in train_dataset:
#  print(item["label"], item["text"])

#for i in range(len(train_dataset['text'])):
#  print(train_dataset['label'][i], train_dataset['text'][i])

# Print to check data
train_dataset_print = train_dataset[:50]

for label, text in zip(train_dataset_print['label'], train_dataset_print['text']):
    print(label, text)


## Take random examples
small_irony_dataset = DatasetDict(
    train = train_dataset.shuffle(seed=24).select(range(128)),
    val = validation_dataset.shuffle(seed=24).select(range(32)),
    test = test_dataset.shuffle(seed=24).select(range(32)))

# *** IMPORT GOOGLE DRIVE

# from google.colab import drive
# drive.mount('/content/drive')

# *** roBERTa

tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")

set_seed(24) ## global seed

## Tokenize

def tokenize_function(examples):
    return tokenizer(examples["text"], padding=True, truncation=True)

small_tokenized_dataset = small_irony_dataset.map(tokenize_function, batched=True, batch_size=16)
data_collator = DataCollatorWithPadding(tokenizer = tokenizer)

model = AutoModelForSequenceClassification.from_pretrained("FacebookAI/roberta-base", num_labels=2)
accuracy = evaluate.load("accuracy")

arguments = TrainingArguments(
    output_dir="sample_cl_trainer",
    per_device_train_batch_size=16, # original 16 # can change to 8, 16 or 32
    per_device_eval_batch_size=16, # original 16
    logging_steps=8,
    num_train_epochs= 5, # original 5
    eval_strategy="epoch", # run validation at the end of each epoch
    save_strategy="epoch",
    learning_rate= 2e-5, # original 2e-5 # can change to 3 or 4 or 5e-5
    weight_decay=0.01,
    load_best_model_at_end=True,
    report_to='none',
    seed=224 ## local seed
)


def compute_metrics(eval_pred): # Called at the end of validation, gives accuracy
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=predictions, references=labels)


trainer = Trainer(
    model=model,
    args=arguments,
    train_dataset=small_tokenized_dataset['train'],
    eval_dataset=small_tokenized_dataset['val'], # val #change to test when you do your final evaluation!
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# *** TRAIN

trainer.train()

# ***
test_str1 = "@user @user @user @user I'm off to visit great-nephew very ill, only 20 fair"
test_str2 = "Halfway thorough my workday ... Woooo"


# 1 = irony
# 0 = non_irony

fine_tuned_model = AutoModelForSequenceClassification.from_pretrained("sample_cl_trainer/checkpoint-24")
model_inputs = tokenizer(test_str1, return_tensors="pt")
prediction = torch.argmax(fine_tuned_model(**model_inputs).logits)
print(["non-irony", "irony"][prediction])

# *** SAVE MODEL

# save_directory = "/content/drive/MyDrive/Colab Notebooks/Project_Irony_Detection/test"
# tokenizer.save_pretrained(save_directory)
# model.save_pretrained(save_directory)