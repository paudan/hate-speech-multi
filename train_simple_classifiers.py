import os
import sys
import logging
import pandas as pd
import torch
from dataset.berkeley_dataset import create_agg_multitask_dataset
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

model_path = "intfloat/e5-small-v2"
cache_dir = 'cache'
output_dir = 'outputs'
if not os.path.exists():
    os.mkdir(output_dir)
file_path = 'measuring-hate-speech.csv'

agg_dataset = create_agg_multitask_dataset(file_path)
tasks_finished = {'religion', 'race', 'gender', 'origin'}
tasks = set(agg_dataset.columns.tolist()) - {'text'}
all_results = list()
tasks_selected = tasks - tasks_finished
for target in tasks_selected:
    logging.info(f"Started training classifier for target {target}")
    _, results = train_eval_model(
        model_path, 
        inputs=agg_dataset['text'].tolist(), 
        targets=agg_dataset[target].tolist(),
        tuned_layers_count=0,
        cache_dir=cache_dir, 
        save_model_dir=os.path.join(output_dir, "berkeley_classifier_" + target), 
        batch_size=64, 
        num_epochs=20, 
        eval_batch_size=1024,
        task_name=target
    )
    results['target'] = target
    all_results.append(results)
all_results = pd.DataFrame(all_results)
print(all_results)
all_results.to_csv('single_classifier_results.csv', index=None)