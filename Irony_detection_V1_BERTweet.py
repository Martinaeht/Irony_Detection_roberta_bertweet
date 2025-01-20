!pip install emoji==0.6.0

#tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=False)

set_seed(24) ## global seed

## Tokenize

def tokenize_function(examples):
    return tokenizer(examples["text"], padding=True)

small_tokenized_dataset = small_irony_dataset.map(tokenize_function, batched=True, batch_size=16)
data_collator = DataCollatorWithPadding(tokenizer = tokenizer)

model = AutoModelForSequenceClassification.from_pretrained("vinai/bertweet-base", num_labels=2)
accuracy = evaluate.load("accuracy")

arguments = TrainingArguments(
    output_dir="sample_cl_trainer_BERTweet",
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



########


#to test
print(small_tokenized_dataset['train'][2])

########

trainer.train()

##########
#testing

test_str1 = "@user @user @user @user I'm off to visit great-nephew very ill, only 20 fair"
test_str2 = "Halfway thorough my workday ... Woooo"


# 1 = irony
# 0 = non_irony

fine_tuned_model = AutoModelForSequenceClassification.from_pretrained("sample_cl_trainer_BERTweet/checkpoint-40")
model_inputs = tokenizer(test_str2, return_tensors="pt")
prediction = torch.argmax(fine_tuned_model(**model_inputs).logits)
print(["non-irony", "irony"][prediction])



#######
# saving pretrained model
save_directory = "/content/drive/MyDrive/Colab Notebooks/Project_Irony_Detection/BERTweet_pretrained"
tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)


#######
#OPTUNA Bertweet


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
        output_dir="sample_cl_trainer_BERTweet_optuna",
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


#######

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)  # You can adjust the number of trials as needed

# Print the best trial
print(f"Best trial: {study.best_trial.params}")


#######
# save optuna results

import os
import json

save_directory = "/content/drive/MyDrive/Colab Notebooks/Project_Irony_Detection/BERTweet_optuna_test"

if not os.path.exists(save_directory):
  os.makedirs(save_directory)

# Save the best trial parameters to a JSON file in the save directory
best_params_path = f"{save_directory}/best_params.json"
with open(best_params_path, 'w') as f:
    json.dump(study.best_trial.params, f)

print(f"Best trial parameters saved to {best_params_path}")

# Save the tokenizer and model to the save directory
tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)


######
# final training with best parameters

# Final Training with optimal hyperparameters
best_params = study.best_trial.params

# Final Training with optimal hyperparameters
arguments = TrainingArguments(
    output_dir="/content/drive/MyDrive/Colab Notebooks/Project_Irony_Detection/BERTweet_optuna_model_trainer",
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

 #######


