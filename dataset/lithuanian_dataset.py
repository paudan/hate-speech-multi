#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Paulius Danėnas"
__copyright__ = "Copyright 2025, Paulius Danėnas, Kaunas University of Technology"
__maintainer__ = "Paulius Danėnas"
__email__ = "danpaulius@gmail.com"


import os
import shutil
import glob
import itertools
from pathlib import Path
import pandas as pd
from tqdm import tqdm
try:
    from .utils import split_data_file
except ImportError:
    from utils import split_data_file


def create_long_format_dataset(file_path, output_path, additional_data: str=None, dataset_name="lith_dataset",
                               create_false_entries=False, balance_dataset=False, sample_ratio=1):
    if isinstance(output_path, str):
        output_path = Path(output_path)
    shutil.rmtree(output_path/dataset_name, ignore_errors=True)
    shutil.rmtree(output_path/f"{dataset_name}_multi", ignore_errors=True)
    final_df = pd.concat([
        pd.read_excel(file_path, sheet_name='Hate', header=0, usecols=['Comment', 'Target']).assign(Value=1),
        pd.read_excel(file_path, sheet_name='Neutralūs', header=0, usecols=['Comment', 'Target']).assign(Value=0)
    ])
    final_df.columns = ['text', 'target', 'value']
    if additional_data is not None:
        add_df = pd.read_csv(additional_data)
        final_df = pd.concat([final_df, add_df])
    # Normalize name and set target to Other when it is not available
    final_df['target'] = final_df['target'].str.replace(' ', '-')
    final_df['target'] = final_df['target'].fillna('Other')
    # Remove cases when targets do not have sufficient sample
    counts = final_df['target'].value_counts()
    selected = counts[counts > 5].index.tolist()
    final_df = final_df[final_df['target'].isin(selected)]
    # Create instances which have 0 value 
    if create_false_entries is True:
        new_df = list(itertools.product(final_df['text'].unique(), final_df['target'].unique()))
        new_df = pd.DataFrame(new_df, columns=['text', 'target']).assign(value=0)
        selected = final_df[(final_df['value'] != 0)]
        for _, row in selected.iterrows():
            new_df.loc[((new_df['text'] == row['text']) & (new_df['target'] == row['target'])), 'value'] = row['value']
        final_df = new_df
    final_df['task_name'] = final_df['target']  # Preserve task name column
    final_df.to_parquet(output_path/dataset_name, partition_cols="target")
    parquet_files = glob.glob(f"{str(output_path)}/{dataset_name}/**/*.parquet", recursive=True)
    for split_file in tqdm(parquet_files):
        group_name = split_file.split(os.path.sep)[-2]
        print(group_name)
        try:
            final_dataset = split_data_file(str(split_file), class_col='value', balance_dataset=balance_dataset, sample_ratio=sample_ratio)
            final_dataset.save_to_disk(os.path.join(str(output_path), f"{dataset_name}_multi", group_name))
        except ValueError as e:
            print(f"Error while processing group {group_name}, skipping: {e}")


def create_wide_format_dataset(file_path, output_path, fill_value=0):
    output_fname = "lith_dataset_multi"
    output_file = os.path.join(output_path, f"{output_fname}.parquet")
    final_df = pd.concat([
        pd.read_excel(file_path, sheet_name='Hate', header=0, usecols=['Comment', 'Target', 'Ar dublis?']).assign(Value=1),
        pd.read_excel(file_path, sheet_name='Neutralūs', header=0, usecols=['Comment', 'Target', 'Ar dublis?']).assign(Value=0)
    ])
    final_df.columns = ['text', 'target', 'duplicate', 'value']
    # Normalize name and set target to Other when it is not available
    final_df['target'] = final_df['target'].str.replace(' ', '-')
    final_df['target'] = final_df['target'].fillna('Other')
    wide_df = pd.pivot(final_df.drop_duplicates(['text']), columns='target', values='value', index='text').reset_index()
    for ind, row in final_df[final_df['duplicate'] == 1].iterrows():
        wide_df.loc[wide_df['text'] == row['text'], row['target']] = row['value']
    # Set False values where NA
    if fill_value:
        wide_df = wide_df.fillna(fill_value)
    wide_df.to_parquet(output_file)
    wide_df.to_csv(os.path.join(output_path, f"{output_fname}.csv"), index=None)
    wide_df = split_data_file(output_file)
    wide_df.save_to_disk(os.path.join(output_path, "lith_dataset_multi_wide"))


if __name__ == '__main__':
    file_path = Path("../data")/'DATASET No. 1 Ethnicity _ nationality _ race.xlsx'
    output_path = 'data'
    create_long_format_dataset(file_path, output_path)
    create_long_format_dataset(file_path, output_path, 
        additional_data=Path("../augmentation")/'lthate'/'generated.csv', 
        dataset_name="lith_dataset_gen"
    )
    create_long_format_dataset(file_path, output_path, 
        additional_data=None, 
        dataset_name="lith_dataset_balanced",
        create_false_entries=True,
        balance_dataset=True,
        sample_ratio=1.5
    )
    create_wide_format_dataset(file_path, output_path)