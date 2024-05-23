#!/usr/bin/env python
# coding: utf-8

# In[3]:
import torch
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('name')
parser.add_argument('transformer_path')
parser.add_argument('normalization_rule')
parser.add_argument('--tokenizer_path', default=None)

args = parser.parse_args()

from transformers import set_seed, TrainingArguments, IntervalStrategy
import sys
sys.path.insert(1, '/path/to/glaux/nlp')
#for reproducibility:
set_seed(1234)
import os
# os.environ['WANDB_PROJECT'] = 'ML4AL'
from classification.Classifier import Classifier

base_output_dir = 'output_dir'

train_path = '../final_dataset/reduced/train.conll'

test_path = '../final_dataset/reduced/test.conll'

val_path = '../final_dataset/reduced/val.conll'

if not args.tokenizer_path:
    tokenizer = args.transformer_path
else:
    tokenizer = args.tokenizer_path

clrf = Classifier(args.transformer_path,base_output_dir + args.name,
                  tokenizer_path=tokenizer,
                  training_data= train_path,
                  eval_data=val_path,
                  test_data=test_path,
                  ignore_label='O', data_preset='simple', add_prefix_space=True, monitor_metric='seqeval_ner')


# In[4]:


from data import Datasets
from tokenization import Tokenization


train_tokens, train_tags = clrf.reader.read_tags('MISC', clrf.training_data, in_feats=False,return_wids=False)
train_tag_dict = {'MISC':train_tags}

train_tokens_norm = Tokenization.normalize_tokens(train_tokens, args.normalization_rule)

tag2id, id2tag = clrf.id_label_mappings(train_tags)
training_data = Datasets.build_dataset(train_tokens_norm,train_tag_dict)
training_data = training_data.map(Tokenization.tokenize_sentence,fn_kwargs={"tokenizer":clrf.tokenizer})
training_data = training_data.map(clrf.align_labels, fn_kwargs={"tag2id":tag2id})
print(type(training_data))
print(f'training with {len(training_data)} samples')

eval_tokens, eval_tags = clrf.reader.read_tags('MISC', clrf.eval_data, in_feats=False,return_wids=False)
eval_tag_dict = {'MISC':eval_tags}
eval_tokens_norm = Tokenization.normalize_tokens(eval_tokens, args.normalization_rule)

eval_data = Datasets.build_dataset(eval_tokens_norm,eval_tag_dict)
eval_data = eval_data.map(Tokenization.tokenize_sentence,fn_kwargs={"tokenizer":clrf.tokenizer})
eval_data = eval_data.map(clrf.align_labels,fn_kwargs={"tag2id":tag2id})
print(f'eval_data with {len(eval_data)} samples')


def wandb_hp_space(trial):
    return {
        "method": "random",
        "metric": {"name": "eval_f1", "goal": "maximize"},
        "parameters": {
            "learning_rate": {"distribution": "uniform", "min": 1e-6, "max": 1e-4},
            "weight_decay": {"values": [0.1, 0.01, 0.001]},
            "num_train_epochs": {"values": [3,4, 5,6]}
        },
    }

# In[5]:

# set training arguments
training_args = TrainingArguments(output_dir=clrf.model_dir,
                                  report_to = 'wandb',
                                  num_train_epochs=5,
                                  per_device_train_batch_size=16,
                                  per_device_eval_batch_size=16,
                                  learning_rate=2e-5, 
                                  save_strategy= IntervalStrategy.STEPS,
                                  evaluation_strategy = IntervalStrategy.STEPS, # "steps"
                                  eval_steps = 500, # Evaluation and Save happens every 500 steps, also default for when training loss is
                                  #reported so seemed appropriate
                                  save_total_limit = 1, # Only last 3 models are saved. Older ones are deleted.
                                  weight_decay=0.01,
                                  push_to_hub=False,
                                  metric_for_best_model = 'f1',
                                  load_best_model_at_end=True,
                                  warmup_ratio = 0.1,
                                  run_name = args.name
                                  )


clrf.train_classifier(output_model=clrf.model_dir, train_dataset=training_data ,eval_dataset=eval_data,tag2id=tag2id,id2tag=id2tag, training_args=training_args, do_hypopt=wandb_hp_space)


# %%
