# -*- coding: utf-8 -*-

__author__ = "Paulius Danėnas"
__copyright__ = "Copyright 2025, Paulius Danėnas, Kaunas University of Technology"
__maintainer__ = "Paulius Danėnas"
__email__ = "danpaulius@gmail.com"


import os
import glob
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, DatasetDict, concatenate_datasets
from sklearn.utils import compute_class_weight

def create_balanced_dataset(dataset, class_col, sample_ratio=1):
    unique, counts = np.unique(dataset[class_col], return_counts=True)
    subsets = []
    sample_size = min(counts)
    minority_class = np.argmin(counts)
    for ind, val in enumerate(counts):
        subset = dataset.filter(lambda x: x[class_col] == unique[ind])
        if ind != minority_class:
            sample_size = np.ceil(sample_size * sample_ratio).astype(int)
        sampled = np.random.choice(range(val), size=sample_size)
        subset = subset.select(sampled)
        subsets.append(subset)
    sampled_dataset = concatenate_datasets(subsets)
    return sampled_dataset

def split_data_file(data_file, class_col=None, train_size=0.7, balance_dataset=False, sample_ratio=1):
    dataset: DatasetDict = load_dataset("parquet", data_files=[data_file], cache_dir='cache')
    train_split = dataset['train']
    if balance_dataset:
        train_split = create_balanced_dataset(train_split, class_col, sample_ratio)    
    if class_col:
        train_split = train_split.class_encode_column(class_col)
    splits = train_split.train_test_split(train_size=train_size, stratify_by_column=class_col)
    splits_val = splits['test'].train_test_split(train_size=0.5, stratify_by_column=class_col)
    return DatasetDict({
        'train': splits['train'],
        'validation': splits_val['train'],
        'test': splits_val['test']
    })

def process_splits(final_df, output_path, dataset_name, balance_dataset=False, sample_ratio=1):
    final_df['task_name'] = final_df['target']  # Preserve task name column
    all_tasks = final_df['task_name'].unique()
    final_df.to_parquet(output_path/dataset_name, partition_cols="target")
    parquet_files = glob.glob(f"{str(output_path)}/{dataset_name}/**/*.parquet", recursive=True)
    for split_file in tqdm(parquet_files):
        group_name = split_file.split(os.path.sep)[-2]
        if group_name.split("=")[1] in all_tasks:
            print(group_name)
            try:
                final_dataset = split_data_file(str(split_file), class_col='value', balance_dataset=balance_dataset, sample_ratio=sample_ratio)
                final_dataset.save_to_disk(os.path.join(str(output_path), f"{dataset_name}_multi", group_name))
            except ValueError as e:
                print(f"Error while processing group {group_name}, skipping: {e}")

def load_complete_dataset(dataset_dir):
    dataset = DatasetDict.load_from_disk(dataset_dir)
    return concatenate_datasets(dataset.values())

def calculate_class_weights(dataset_dir):
    data = load_complete_dataset(dataset_dir)
    label_columns = list(set(data.features.keys()) - {'text'})
    weights = {}
    for column in label_columns:
        weights[column] = compute_class_weight(class_weight='balanced', classes=np.unique(data[column]), y=data[column])
    return weights