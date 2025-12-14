#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Paulius Danėnas"
__copyright__ = "Copyright 2025, Paulius Danėnas, Kaunas University of Technology"
__maintainer__ = "Paulius Danėnas"
__email__ = "danpaulius@gmail.com"


import glob
import itertools
import json
import os
import shutil
from pathlib import Path
import pandas as pd
from tqdm import tqdm
try:
    from .utils import split_data_file
except ImportError:
    from utils import split_data_file


def _create_dataset(input_file):
    with open(input_file, 'r', encoding='utf8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df['types'] = df['annotations'].apply(lambda x: list(set(t['technique'] for t in x)) if x and isinstance(x, list) else [])
    available_types = set(itertools.chain.from_iterable(df['types'].values.tolist()))
    final_df = pd.concat([
        df[['comment_id', 'content']].copy().assign(
            target='Contains manipulation',
            value=(df['found_manipulation'] == 'yes').astype(int)
        ).rename(columns={'content': 'text'}),
        pd.concat(df.apply(
            lambda x: pd.DataFrame(data=[{
                'comment_id': x['comment_id'], 
                'text': x['content'], 
                'target': t, 
                'value': 1 if t in x['types'] else 0
            } for t in available_types]), 
            axis=1
        ).values.tolist())
    ]).reset_index(drop=True)
    return final_df

def create_binary_dataset(input_file):
    with open(input_file, 'r', encoding='utf8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    return df[['content']].assign(target=(df['found_manipulation'] == 'yes').astype(int))

def create_wide_format_dataset(input_file, output_path, dataset_name="aicp_fimi_dataset_multi"):
    output_file = os.path.join(output_path, f"{dataset_name}.parquet")
    final_dataset = _create_dataset(input_file)
    final_dataset = final_dataset.pivot(columns='target', index=['comment_id', 'text'], values='value').reset_index()
    final_dataset = final_dataset.fillna(value=0)
    final_dataset.to_parquet(output_file)
    final_dataset.to_csv(os.path.join(output_path, f"{dataset_name}_wide.csv"), index=None)
    final_dataset = split_data_file(output_file)
    final_dataset.save_to_disk(os.path.join(output_path, dataset_name + "_wide"))

def create_long_format_dataset(input_file, output_path, dataset_name="aicp_fimi_dataset"):
    if isinstance(output_path, str):
        output_path = Path(output_path)
    shutil.rmtree(output_path/dataset_name, ignore_errors=True)
    shutil.rmtree(output_path/f"{dataset_name}_multi", ignore_errors=True)
    final_df = _create_dataset(input_file)
    final_df['task_name'] = final_df['target']  # Preserve task name column
    final_df.to_parquet(output_path/dataset_name, partition_cols="target")
    parquet_files = glob.glob(f"{str(output_path)}/{dataset_name}/**/*.parquet", recursive=True)
    for split_file in tqdm(parquet_files):
        group_name = split_file.split(os.path.sep)[-2]
        final_dataset = split_data_file(str(split_file), class_col='value')
        final_dataset.save_to_disk(os.path.join(str(output_path), f"{dataset_name}_multi", group_name))


if __name__ == '__main__':
    input_file = '../data/corpus_AICP_FIMI.json'
    output_path = 'data'
    create_wide_format_dataset(input_file, output_path)
    create_long_format_dataset(input_file, output_path)