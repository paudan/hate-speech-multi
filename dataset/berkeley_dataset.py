#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Paulius Danėnas"
__copyright__ = "Copyright 2025, Paulius Danėnas, Kaunas University of Technology"
__maintainer__ = "Paulius Danėnas"
__email__ = "danpaulius@gmail.com"


import glob
import os
import shutil
from pathlib import Path
import datasets
import pandas as pd
from tqdm import tqdm
try:
    from .utils import split_data_file
except ImportError:
    from utils import split_data_file


def create_agg_multitask_dataset():
    dataset = datasets.load_dataset('ucberkeley-dlab/measuring-hate-speech')   
    data = dataset['train'].to_pandas()
    target_columns = list(filter(lambda x: x.startswith('target'), data.columns.tolist()))
    # As we have multiple annotators, use majority voting to define final labels
    additional_columns = ['hatespeech']
    core_columns = ['text', 'comment_id']
    agg_data = data[['comment_id'] + target_columns + additional_columns].groupby('comment_id')\
        .agg(lambda x: max(set(x), key=x.tolist().count))\
        .reset_index()
    agg_data = data[core_columns].drop_duplicates().merge(agg_data, on='comment_id')
    grouped_columns = pd.DataFrame(data={'column': target_columns})
    grouped_columns['group'] = grouped_columns['column'].apply(lambda x: x.split("_")[1] if len(x.split("_")) > 1 else x)
    grouped_columns = grouped_columns.groupby('group')['column'].agg(list).to_dict()
    final_df = agg_data[core_columns].copy()
    for agg_col, aggregates in grouped_columns.items():
        final_df[agg_col] = agg_data[aggregates].sum(axis=1)
    for agg_col in additional_columns:
        final_df[agg_col] = agg_data[agg_col].astype(int)
    agg_cols = list(grouped_columns.keys())
    final_df[agg_cols] = (final_df[agg_cols] > 0).astype(int)
    # Calculate hate speech target column as general indication
    final_df['contains_hate'] = (final_df[agg_cols].sum(axis=1) > 0).astype(int)
    # Discard non-relevant cols
    final_df = final_df.drop(labels='comment_id', axis=1)
    return final_df


def create_wide_format_dataset(output_path):
    output_file = os.path.join(output_path, "berkeley_dataset_multi.parquet")
    final_dataset = create_agg_multitask_dataset()
    final_dataset.to_parquet(output_file)
    final_dataset.to_csv(os.path.join(output_path, "berkeley_dataset_multi.csv"), index=None)
    final_dataset = split_data_file(output_file)
    final_dataset.save_to_disk(os.path.join(output_path, "hf_berkeley_multi_wide"))


def create_long_format_dataset(output_path, dataset_name="berkeley_dataset", additional_data: str=None):
    if isinstance(output_path, str):
        output_path = Path(output_path)
    shutil.rmtree(output_path/dataset_name, ignore_errors=True)
    shutil.rmtree(output_path/f"{dataset_name}_multi", ignore_errors=True)
    dataset = create_agg_multitask_dataset()
    final_df = pd.melt(dataset, id_vars='text', var_name='target')
    if additional_data is not None:
        add_df = pd.read_csv(additional_data)
        final_df = pd.concat([final_df, add_df])
    final_df['task_name'] = final_df['target']  # Preserve task name column
    final_df.to_parquet(output_path/dataset_name, partition_cols="target")
    parquet_files = glob.glob(f"{str(output_path)}/{dataset_name}/**/*.parquet", recursive=True)
    for split_file in tqdm(parquet_files):
        group_name = split_file.split(os.path.sep)[-2]
        final_dataset = split_data_file(str(split_file), class_col='value')
        final_dataset.save_to_disk(os.path.join(str(output_path), f"{dataset_name}_multi", group_name))


if __name__ == '__main__':
    output_path = 'data'
    create_wide_format_dataset(output_path)
    create_long_format_dataset(output_path)
