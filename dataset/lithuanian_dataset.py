#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Paulius Danėnas"
__copyright__ = "Copyright 2025, Paulius Danėnas, Kaunas University of Technology"
__maintainer__ = "Paulius Danėnas"
__email__ = "danpaulius@gmail.com"


import os
import shutil
import itertools
from pathlib import Path
import pandas as pd
try:
    from .utils import split_data_file, process_splits
except ImportError:
    from utils import split_data_file, process_splits


INPUT_FILES_RACE = [
    ('DATASET No. 1 Ethnicity _ nationality _ race_HS.v1.csv', 1),
    ('DATASET No. 1 Ethnicity _ nationality _ race_N.v1.csv', 0),
]
INPUT_FILES_ORIENTATION = [
    ('DATASET No. 2 S. Orientation _ gender_HS.v1.csv', 1),
    ('DATASET No. 2 S. Orientation _ gender_N.v1.csv', 0)
]
INPUT_FILES_COUNTRIES = [
    ('DATASET No. 3 Countries_HS.v1.csv', 1),
    ('DATASET No. 3 Countries_N.v1.csv', 0)
]
INPUT_FILES_POLITICAL = [
    ('DATASET No. 4 Political_viewa_N.v1.csv', 0),
    ('DATASET No. 4 Political_views_HS.v1.csv', 1)
]


def preprocess_targets(final_df: pd.DataFrame):
    # Normalize name and set target to Other when it is not available
    final_df['target'] = final_df['target'].str.replace(' ', '-')
    final_df['target'] = final_df['target'].fillna('Other')
    # Remove cases when targets do not have sufficient sample
    counts = final_df['target'].value_counts()
    selected = counts[counts > 5].index.tolist()
    final_df = final_df[final_df['target'].isin(selected)]
    return final_df


def _create_long_format_dataset_hate(
        data_path, data_inputs, output_path, dataset_name="lith_dataset", additional_data: str=None, 
        create_false_entries=False, balance_dataset=False, sample_ratio=1, clear_dirs=True
):
    if isinstance(output_path, str):
        output_path = Path(output_path)
    if isinstance(data_path, str):
        data_path = Path(data_path)
    if clear_dirs:
        shutil.rmtree(output_path/dataset_name, ignore_errors=True)
        shutil.rmtree(output_path/f"{dataset_name}_multi", ignore_errors=True)
    read_file = lambda x: pd.read_csv(data_path/x[0], sep=';', header=0, usecols=['Comment', 'Target']).assign(Value=x[1])
    final_df = pd.concat(list(map(read_file, data_inputs)))
    final_df.columns = ['text', 'target', 'value']
    if additional_data is not None:
        add_df = pd.read_csv(additional_data)
        final_df = pd.concat([final_df, add_df])
    final_df = preprocess_targets(final_df)
    # Create instances which have 0 value 
    if create_false_entries is True:
        new_df = list(itertools.product(final_df['text'].unique(), final_df['target'].unique()))
        new_df = pd.DataFrame(new_df, columns=['text', 'target']).assign(value=0)
        selected = final_df[(final_df['value'] != 0)]
        for _, row in selected.iterrows():
            new_df.loc[((new_df['text'] == row['text']) & (new_df['target'] == row['target'])), 'value'] = row['value']
        final_df = new_df
    process_splits(final_df, output_path, dataset_name, balance_dataset=balance_dataset, sample_ratio=sample_ratio)


def _create_long_format_dataset_levels(
        data_path, data_inputs, output_path, dataset_name="lith_dataset", 
        balance_dataset=False, sample_ratio=1, clear_dirs=True
):
    if isinstance(output_path, str):
        output_path = Path(output_path)
    if isinstance(data_path, str):
        data_path = Path(data_path)
    if clear_dirs:
        shutil.rmtree(output_path/dataset_name, ignore_errors=True)
        shutil.rmtree(output_path/f"{dataset_name}_multi", ignore_errors=True)

    def read_file_levels(input):
        usecols = ['Comment', 'Target']
        if input[1] == 1:
            usecols.append('HS Level')
        data = pd.read_csv(data_path/input[0], sep=';', header=0, usecols=usecols)
        if input[1] == 0:
            data = data.assign(value='Level 0')
        elif input[1] == 1:
            data = data.rename(columns={'HS Level': 'value'})
        return data
        
    levels_df = pd.concat(list(map(read_file_levels, data_inputs)))
    levels_df.columns = ['text', 'target', 'value']
    levels_df = preprocess_targets(levels_df)
    levels_df['target'] = levels_df['target'].apply(lambda x: x + "-Level")
    levels_df['value'] = levels_df['value'].astype(str)
    process_splits(levels_df, output_path, dataset_name, balance_dataset=balance_dataset, sample_ratio=sample_ratio)


def create_long_format_dataset(
        file_path, data_inputs, output_path, dataset_name="lith_dataset", 
        additional_data: str=None, create_false_entries=False, balance_dataset=False, 
        sample_ratio=1, add_levels_data=False
):
    _create_long_format_dataset_hate(
        file_path, data_inputs, output_path, dataset_name, additional_data, 
        create_false_entries, balance_dataset, sample_ratio, clear_dirs=True
    )
    if add_levels_data:
        _create_long_format_dataset_levels(
            file_path, data_inputs, output_path, dataset_name=dataset_name, 
            balance_dataset=False, sample_ratio=sample_ratio, clear_dirs=False
        )


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
    data_path = Path("../data")
    output_path = 'data'
    # create_long_format_dataset(file_path, output_path, 
    #     additional_data=Path("../augmentation")/'lthate'/'generated.csv', 
    #     dataset_name="lith_dataset_gen"
    # )
    NAMES_MAP = [
        (INPUT_FILES_RACE, 'race'),
        (INPUT_FILES_ORIENTATION, 'orientation'),
        (INPUT_FILES_COUNTRIES, 'countries'),
        (INPUT_FILES_POLITICAL, 'political')
    ]
    for data_inputs, name in NAMES_MAP:
        create_long_format_dataset(
            data_path, 
            data_inputs=data_inputs, 
            output_path=output_path,
            dataset_name=f"lith_dataset_{name}"
        )
        create_long_format_dataset(
            data_path, 
            data_inputs=data_inputs, 
            output_path=output_path, 
            additional_data=None, 
            dataset_name=f"lith_dataset_balanced_{name}",
            create_false_entries=True,
            balance_dataset=True,
            sample_ratio=1.5
        )
        create_long_format_dataset(
            data_path, 
            data_inputs=data_inputs, 
            output_path=output_path, 
            additional_data=None, 
            dataset_name=f"lith_dataset_levels_{name}",
            create_false_entries=True,
            balance_dataset=True,
            sample_ratio=1.5,
            add_levels_data=True
        )
    # create_wide_format_dataset(data_path, output_path)