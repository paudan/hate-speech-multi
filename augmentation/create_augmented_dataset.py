import sys
import os
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import datasets
from datasets import DatasetDict, Dataset

output_dir = "gemma"
data_dir = '../hate-speech-multi/data/2026-01-30'
dataset_path = "../data"
recs_selected = 1

def load_data(result_file, src_file):
    df_generated = pd.read_json(result_file)
    if 'Index' in df_generated.columns:
        df_generated['index'] = np.where(df_generated['index'].isnull(), df_generated['Index'], df_generated['index'])
    if 'Labels' in df_generated.columns:
        df_generated['labels'] = np.where(df_generated['labels'].isnull(), df_generated['Labels'], df_generated['labels'])
    if 'Original_Comment' in df_generated.columns:
        df_generated['original_comment'] = np.where(df_generated['original_comment'].isnull(), df_generated['Original_Comment'], df_generated['original_comment'])
    if 'Generated_Comments' in df_generated.columns:
        df_generated['generated_comments'] = np.where(df_generated['generated_comments'].isnull(), df_generated['Generated_Comments'], df_generated['generated_comments'])
    data = pd.read_csv(src_file)
    labels = set(data.columns.tolist()) - {'text'}
    data['labels'] = data.apply(lambda x: [col for col in labels if x[col] == 1], axis=1)
    items = data[['text', 'labels']].to_dict(orient='records')
    items = list(filter(lambda x: len(x['labels']) > 0, items))
    items = pd.DataFrame(items)
    df_generated = df_generated.drop(labels='labels', axis=1).merge(items, left_on='original_comment', right_on='text')
    df_generated['selected_comment'] = df_generated['generated_comments'].apply(lambda c: list(filter(lambda x: 'šūd' not in x.lower(), c)) if c is not None and isinstance(c, list) else None)
    df_generated['selected_comment'] = df_generated['generated_comments'].apply(lambda c: c[:recs_selected] if isinstance(c, list) and len(c) > 0 else [])
    df_generated = df_generated.explode('selected_comment').explode('labels')
    return df_generated[['index', 'original_comment', 'generated_comments', 'selected_comment', 'labels']]

def create_gen_file(json_output, wide_dataset_file, output_file):
    df_generated = load_data(json_output, os.path.join(data_dir, wide_dataset_file))
    df_generated[['selected_comment', 'labels']]\
        .rename(columns={'selected_comment': 'text', 'labels': 'target'})\
        .assign(value=1)\
        .to_csv(os.path.join(output_dir, output_file), index=None)    

target_groups = ['countries', 'orientation', 'political', 'race', 'religion']
os.makedirs(output_dir, exist_ok=True)
for x in [(f'lith_dataset_wide_{name}_generated.json', f'lith_dataset_wide_{name}.csv', f'hs_{name}_generated.csv') for name in target_groups]:
    create_gen_file(*x)
pd.concat([pd.read_csv(os.path.join(output_dir, f'hs_{name}_generated.csv')) for name in target_groups])\
    .to_csv(os.path.join(output_dir, f'hs_full_generated.csv'))

data_path = Path(dataset_path)
output_path = 'augmented_data_2'

def augment_dataset(data_dir, new_dir, name):
    shutil.rmtree(new_dir, ignore_errors=True)
    shutil.copytree(data_dir, new_dir, dirs_exist_ok=True)
    augmented_data = pd.read_csv(os.path.join(output_dir, f"hs_{name}_generated.csv"))
    groups = augmented_data['target'].unique()
    for group_name in groups:
        task_dir = f'target={group_name}'
        try:
            dataset = DatasetDict.load_from_disk(os.path.join(data_dir, task_dir))
        except FileNotFoundError:
            print(f"Group {group_name} not found in the source dataset, skipping")
            continue
        df_aug = augmented_data[augmented_data['target'] == group_name]
        df_aug = pd.DataFrame({'text': df_aug['text'], 'value': 1, 'task_name': group_name})
        df_aug = Dataset.from_pandas(df_aug).cast_column('value', dataset['train'].features['value'])
        dataset['train'] = datasets.concatenate_datasets([dataset['train'], df_aug])
        dataset.save_to_disk(os.path.join(new_dir, task_dir))

for name in target_groups + ['full']:
    augment_dataset(
        os.path.join(data_dir, f'lith_dataset_balanced_{name}_multi'), 
        os.path.join(output_path, f"lith_dataset_augmented_{name}"),
        name
    )

### If augmented data is injected before split operation ###

# sys.path.append('../hate-speech-multi')
# from dataset.lithuanian_dataset import (
#     create_long_format_dataset, 
#     INPUT_FILES_RACE, INPUT_FILES_ORIENTATION, INPUT_FILES_COUNTRIES, INPUT_FILES_POLITICAL, INPUT_FILES_RELIGION
# )
# NAMES_MAP = [
#     (INPUT_FILES_RACE, 'race'),
#     (INPUT_FILES_ORIENTATION, 'orientation'),
#     (INPUT_FILES_COUNTRIES, 'countries'),
#     (INPUT_FILES_POLITICAL, 'political'),
#     (INPUT_FILES_RELIGION, 'religion')
# ]
# for data_inputs, name in NAMES_MAP:
#     create_long_format_dataset(
#         data_path, 
#         data_inputs=data_inputs, 
#         output_path=output_path, 
#         additional_data=os.path.join(output_dir, f"hs_{name}_generated.csv"), 
#         dataset_name=f"lith_dataset_augmented_{name}",
#         create_false_entries=True,
#         balance_dataset=True,
#         sample_ratio=1.5
#     )
