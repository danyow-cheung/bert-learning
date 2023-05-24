from transformers import BertForSequenceClassification,BertTokenizerFast,Trainer,TrainingArguments
from nlp import load_dataset 
import torch
import numpy as np 
import nlp 

'''Load the model and dataset'''
dataset = load_dataset('csv', data_files='./imdbs.csv', split='train')
dataset = dataset.train_test_split(test_size=0.3)


train_set = dataset['train']
test_set = dataset['test']

'''Download and load the model'''
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

def preprocess(data):
    # return tokenizer(data['text'],padding=True,truncation=True)
    return tokenizer(data['text'], padding=True, truncation=True)
train_set = train_set.map(preprocess,batched=True,batch_size=len(train_set))
test_set = test_set.map(preprocess,batched=True,batch_size = len(test_set))
'''
接下來，我們使用 set_format 函數並在我們的數據集中選擇我們需要的列
以及我們需要的格式，如以下代碼所示：
'''
train_set.set_format('torch',columns=['input_ids','attention_mask','label'])
test_set.set_format('torch',columns = ['input_ids','attention_mask','label'])
'''
Train the model
'''
batch_size = 8 
epochs = 2 
warmup_steps = 500
weight_decay = 0.01 
training_args = TrainingArguments(
    output_dir = './results',
    num_train_epochs=epochs,
    per_device_train_batch_size = batch_size,
    per_device_eval_batch_size = batch_size,
    warmup_steps=warmup_steps,
    weight_decay=weight_decay,
    # evaluate_during_training=True,
    logging_dir='./logs'
)
trainer = Trainer(
    model=model,
    args= training_args,
    train_dataset=train_set,
    eval_dataset=test_set
)
trainer.train()
trainer.evaluate()