# -*- coding: utf-8 -*-

import os
import shutil
import glob
from pathlib import Path
import pandas as pd
from tqdm import tqdm
try:
    from .utils import split_data_file
except ImportError:
    from utils import split_data_file


def create_long_format_dataset(file_path, output_path):
    if isinstance(output_path, str):
        output_path = Path(output_path)
    shutil.rmtree(output_path/"lith_dataset", ignore_errors=True)
    shutil.rmtree(output_path/"lith_dataset_multi", ignore_errors=True)
    final_df = pd.concat([
        pd.read_excel(file_path, sheet_name='Hate', header=0, usecols=['Comment', 'Target']).assign(Value=1),
        pd.read_excel(file_path, sheet_name='Neutralūs', header=0, usecols=['Comment', 'Target']).assign(Value=0)
    ])
    final_df.columns = ['text', 'target', 'value']
    # Normalize name and set target to Other when it is not available
    final_df['target'] = final_df['target'].str.replace(' ', '-')
    final_df['target'] = final_df['target'].fillna('Other')
    # Remove cases when targets do not have sufficient sample
    counts = final_df['target'].value_counts()
    selected = counts[counts > 5].index.tolist()
    final_df = final_df[final_df['target'].isin(selected)]
    final_df['task_name'] = final_df['target']  # Preserve task name column
    final_df.to_parquet(output_path/"lith_dataset", partition_cols="target")
    parquet_files = glob.glob(f"{str(output_path)}/lith_dataset/**/*.parquet", recursive=True)
    for split_file in tqdm(parquet_files):
        group_name = split_file.split(os.path.sep)[-2]
        print(group_name)
        try:
            final_dataset = split_data_file(str(split_file), class_col='value')
            final_dataset.save_to_disk(os.path.join(str(output_path), "lith_dataset_multi", group_name))
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
    create_wide_format_dataset(file_path, output_path)