#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Paulius Danėnas"
__copyright__ = "Copyright 2025, Paulius Danėnas, Kaunas University of Technology"
__maintainer__ = "Paulius Danėnas"
__email__ = "danpaulius@gmail.com"


import os
import sys
import logging
import pandas as pd
import torch
from dataset.aicp_fimi_dataset import _get_wide_dataset
from models.multitask_modernbert import SimpleModernBertClassifier
from simple_trainer import train_eval_model

os.environ["CUDA_VISIBLE_DEVICES"]="0"
print("Using GPU:", torch.cuda.is_available())
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

model_path = "VSSA-SDSA/LT-MLKM-modernBERT"
cache_dir = 'cache'
output_dir = 'outputs-mlkm'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
file_path = '../data/corpus_AICP_FIMI.json'

agg_dataset = _get_wide_dataset(file_path)
tasks_finished = []
# tasks = set(agg_dataset.columns.tolist()) - {'text'}
tasks = ['Contains manipulation']
all_results = list()
tasks_selected = set(tasks) - set(tasks_finished)
for target in tasks_selected:
    logging.info(f"Started training classifier for target {target}")
    _, results = train_eval_model(
        model_path, 
        inputs=agg_dataset['text'].tolist(), 
        targets=agg_dataset[target].tolist(),
        model_class=SimpleModernBertClassifier,
        tuned_layers_count=0,
        cache_dir=cache_dir, 
        save_model_dir=os.path.join(output_dir, "mlkm_classifier_" + target), 
        batch_size=64, 
        num_epochs=20, 
        eval_batch_size=1024,
        task_name=target,
        tokenizer_args=dict(padding='max_length', truncation=True, max_length=1024)
    )
    results['target'] = target
    all_results.append(results)
all_results = pd.DataFrame(all_results)
print(all_results)
all_results.to_csv('mlkm_classifier_results.csv', index=None)