#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Paulius Danėnas"
__copyright__ = "Copyright 2025, Paulius Danėnas, Kaunas University of Technology"
__maintainer__ = "Paulius Danėnas"
__email__ = "danpaulius@gmail.com"


import os
import shutil
import glob
from pathlib import Path
import pandas as pd
from datasets import DatasetDict, load_dataset
from tqdm import tqdm
try:
    from .utils import split_data_file
except ImportError:
    from utils import split_data_file


berkeley_mapping = {
    'women': 'gender',
    'trans people': 'sexuality',
    'gay people': 'sexuality',
    'disabled people': 'disability',
    'black people': 'race',
    'Muslims': 'religion',
    'immigrants': 'origin'
}

def create_long_format_dataset(file_path, output_path):
    if isinstance(output_path, str):
        output_path = Path(output_path)
    shutil.rmtree(output_path/"hatecheck_dataset", ignore_errors=True)
    shutil.rmtree(output_path/"hatecheck_dataset_multi", ignore_errors=True)
    data = pd.read_csv(file_path)
    data['target_ident'] = data['target_ident'].map(berkeley_mapping)
    data_df = pd.concat([
        data[['test_case', 'target_ident']].rename(columns={'target_ident': 'task_name'}).assign(value=1),
        data[['test_case']].assign(task_name='contains_hate', value=(data['label_gold'] == 'hateful'))
    ])
    data_df = data_df[~pd.isnull(data_df['task_name'])]
    data_df.columns = ['text', 'target', 'value']
    data_df['task_name'] = data_df['target']
    data_df.to_parquet(output_path/"hatecheck_dataset", partition_cols="target")
    parquet_files = glob.glob(f"{str(output_path)}/hatecheck_dataset/**/*.parquet", recursive=True)
    for split_file in tqdm(parquet_files):
        group_name = split_file.split(os.path.sep)[-2]
        dataset: DatasetDict = load_dataset("parquet", data_files=[str(split_file)], cache_dir='cache')
        dataset.save_to_disk(os.path.join(str(output_path), "hatecheck_dataset_multi", group_name))


def create_wide_format_dataset(file_path, output_path, fill_value=0):
    output_fname = "hatecheck_dataset_multi"
    output_file = os.path.join(output_path, f"{output_fname}.parquet")
    data = pd.read_csv(file_path)
    data['target_ident'] = data['target_ident'].map(berkeley_mapping)
    data_df = pd.concat([
        data[['test_case', 'target_ident']].rename(columns={'target_ident': 'task_name'}).assign(value=1),
        data[['test_case']].assign(task_name='contains_hate', value=(data['label_gold'] == 'hateful'))
    ])
    wide_df = pd.pivot_table(data_df, columns='task_name', values='value', index='test_case', fill_value=fill_value).reset_index()
    wide_df.to_parquet(output_file)
    wide_df.to_csv(os.path.join(output_path, f"{output_fname}.csv"), index=None)
    dataset: DatasetDict = load_dataset("parquet", data_files=[output_file], cache_dir='cache')
    dataset.save_to_disk(os.path.join(output_path, "hatecheck_dataset_multi_wide"))


if __name__ == '__main__':
    file_path = Path("../data")/'hatecheck_final_ACL.csv'
    output_path = 'data'
    create_long_format_dataset(file_path, output_path)
    create_wide_format_dataset(file_path, output_path)