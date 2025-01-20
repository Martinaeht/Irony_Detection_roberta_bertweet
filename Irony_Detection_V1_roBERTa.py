'''Adapted from a COLAB Notebook'''

import datasets
import transformers
import evaluate
import accelerate
import optuna



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

'''***ROBERTA***'''

tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")

set_seed(24) ## global seed

## Tokenize

def tokenize_function(examples):
    return tokenizer(examples["text"], padding=True)

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


######### to test:
print(small_tokenized_dataset['train'][2])


'''TRAIN'''

trainer.train()

####
test_str1 = "@user @user @user @user I'm off to visit great-nephew very ill, only 20 fair"
test_str2 = "Halfway thorough my workday ... Woooo"


# 1 = irony
# 0 = non_irony

fine_tuned_model = AutoModelForSequenceClassification.from_pretrained("sample_cl_trainer/checkpoint-24")
model_inputs = tokenizer(test_str2, return_tensors="pt")
prediction = torch.argmax(fine_tuned_model(**model_inputs).logits)
print(["non-irony", "irony"][prediction])


save_directory = "/content/drive/MyDrive/Colab Notebooks/Project_Irony_Detection/test"

tokenizer.save_pretrained(save_directory)

model.save_pretrained(save_directory)


#######OPTUNA #######

import optuna

# Optuna objective function
def objective(trial):
    # Suggest hyperparameters using trial.suggest_* methods
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-6, 1e-4)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    num_train_epochs = trial.suggest_int('num_train_epochs', 3, 6)
    weight_decay = trial.suggest_uniform('weight_decay', 0.0, 0.1)

    # Training arguments
    arguments = TrainingArguments(
        output_dir="sample_cl_trainer",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        logging_steps=8,
        num_train_epochs=num_train_epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        load_best_model_at_end=True,
        report_to='none',
        seed=224
    )

    # Trainer setup
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=small_tokenized_dataset['train'],
        eval_dataset=small_tokenized_dataset['val'],
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    eval_results = trainer.evaluate()

    # Return the evaluation accuracy as the optimization metric
    return eval_results["eval_accuracy"]

# Create an Optuna study and optimize the objective function
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)  # You can adjust the number of trials as needed

# Print the best trial
print(f"Best trial: {study.best_trial.params}")

######

save_directory = "/content/drive/MyDrive/Colab Notebooks/Project_Irony_Detection/test_optuna"

tokenizer.save_pretrained(save_directory)

model.save_pretrained(save_directory)



# final training with best parameters

# Final Training with optimal hyperparameters
best_params = study.best_trial.params

# Final Training with optimal hyperparameters
arguments = TrainingArguments(
    output_dir="/content/drive/MyDrive/Colab Notebooks/Project_Irony_Detection/final_model_trainer",
    per_device_train_batch_size=best_params['batch_size'],
    per_device_eval_batch_size=best_params['batch_size'],
    logging_steps=8,
    num_train_epochs=best_params['num_train_epochs'],
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=best_params['learning_rate'],
    weight_decay=best_params['weight_decay'],
    load_best_model_at_end=True,
    report_to='none',
    seed=224
)

trainer = Trainer(
    model=model,
    args=arguments,
    train_dataset=small_tokenized_dataset['train'],
    eval_dataset=small_tokenized_dataset['val'],
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()


######TEST ######
#### TO BE ADAPTED  -TEST?!!!!!!!!!!!!!!!!!!!!!!!!!!!!

import os

test_str = "I love this movie!"

output_dir_test = "results_test_set" ## adapt
if not os.path.exists(output_dir_test):
  os.makedirs(output_dir_test)

#adapt checkpoint
fine_tuned_model = AutoModelForSequenceClassification.from_pretrained("sample_cl_trainer/checkpoint-40")

for example in small_imdb_dataset['test']:
  text = example['text']
  model_inputs = tokenizer(text, return_tensors="pt")
  prediction = torch.argmax(fine_tuned_model(**model_inputs).logits)
  #print(example['text'])
  print(["NEGATIVE", "POSITIVE"][prediction])

#for example in small_imdb_dataset['test']:
#  print(example['text'])

arguments_test_set = TrainingArguments(
    output_dir="results_test_set",
    per_device_eval_batch_size=16,
    report_to='none',
    seed=224
)

trainer_test_set = Trainer(
    model=fine_tuned_model,
    args=arguments_test_set,
    eval_dataset=small_tokenized_dataset['test'],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

test_results = trainer_test_set.evaluate(eval_dataset=small_tokenized_dataset['test'])
print(f"Test Accuracy: {test_results['eval_accuracy']}")

##adapt
# output: Test Accuracy: 0.6875, compared to training accuracy: 0.781250


######VISUALIZATION ''' PROBABLY NEEDS TO BE ADAPTED

###JUST THIS

tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
fine_tuned_model = AutoModelForSequenceClassification.from_pretrained("/content/drive/MyDrive/Colab Notebooks/Project_Irony_Detection/final_model_trainer/checkpoint-48") # this has to correspond to the directory in the arguments (where you stored your model)
# visualize last layer of first checkpoint and last layer of last checkpoint

model_inputs = tokenizer(small_tokenized_dataset['val']['text'], padding=True, truncation=True, return_tensors='pt') # run on test (!) set
outputs = fine_tuned_model(**model_inputs, output_hidden_states=True)

#### AND THIS

import torch
import os
# associates reviews with labels

path = "/content/drive/MyDrive/Colab Notebooks/Project_Irony_Detection/results_vis" # stored in: results - layer - 0000 - default  - we need the two tsv files - download - load in project tensor flow (first load tensors, then metadata)
layer=0 # middle layers tend to look better in
if not os.path.exists(path):
  os.mkdir(path)

# if you have a labeled dataset, you can choose to display them in the embedding projector.

while layer in range(len(outputs['hidden_states'])):
  if not os.path.exists(path+'/layer_' + str(layer)):
    os.mkdir(path+'/layer_' + str(layer))

  example = 0
  tensors = []
  labels = []

  while example in range(len(outputs['hidden_states'][layer])):
    #sp_token_position = 0
    #for token in model_inputs['input_ids'][example]:
    #  if token != 101:
    #    sp_token_position += 1
    #  else:
        tensor = outputs['hidden_states'][layer][example][0] # [sp_token_position]
        tensors.append(tensor)
    #    break

        label = [small_tokenized_dataset['val']['text'][example],str(small_tokenized_dataset['val']['label'][example])]
        labels.append(label)
        example +=1

  writer=SummaryWriter(path+'/layer_' + str(layer))
  writer.add_embedding(torch.stack(tensors), metadata=labels, metadata_header=['text','label'])

  layer+=1


