#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Paulius Danėnas"
__copyright__ = "Copyright 2025, Paulius Danėnas, Kaunas University of Technology"
__maintainer__ = "Paulius Danėnas"
__email__ = "danpaulius@gmail.com"


import os
import itertools
from typing import Union
from bidict import bidict
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from datasets import DatasetDict
from .preprocess import preprocess_text
from .utils import load_complete_dataset

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

TOKENIZER_ARGS = dict(
    return_token_type_ids=True, 
    return_attention_mask=True,
    return_tensors='pt', 
    padding='max_length', 
    truncation=True
)


def load_task_dataset(task_dir, split):
    try:
        dataset = DatasetDict.load_from_disk(task_dir)
        dataset_split = dataset[split]
    except Exception as exc:
        print(f"Error while loading task dataset from {task_dir}:", exc.__str__())
        return None
    return dataset_split


class SimpleDataset(Dataset):

    def __init__(self, inputs, targets, preprocessor: Union[PreTrainedTokenizer, PreTrainedTokenizerFast], tokenizer_args={}):
        super().__init__()
        self.data = inputs
        self.labels = targets
        self.preprocessor = preprocessor
        self.class_map = self.class_to_idx(targets)
        self.tokenizer_args = TOKENIZER_ARGS
        if isinstance(tokenizer_args, dict):
            self.tokenizer_args.update(tokenizer_args)

    def class_to_idx(self, instance_labels):
        classes = sorted(set(instance_labels))
        return bidict({j: i for i, j in enumerate(classes)})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        inputs = preprocess_text(self.data[index])
        inputs = self.preprocessor(inputs, **self.tokenizer_args)
        inputs['labels'] = self.class_map[self.labels[index]]
        return inputs


class MultitaskDatasetWide(Dataset):

    def __init__(self, dataset_dir: DatasetDict, preprocessor: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
                 split=None, class_maps=None, load_all_data=False, tokenizer_args={}):
        super().__init__()
        if split is None and load_complete_dataset is False:
            raise ValueError("split parameter must be set if load_complete_dataset is set to False")
        if load_all_data is True:
            self.data = load_complete_dataset(dataset_dir)
        else:    
            self.data = load_task_dataset(dataset_dir, split)
        # self.data = self.data.select(range(1000))
        self.preprocessor = preprocessor
        self.label_columns = list(set(self.data.features.keys()) - {'text'})
        self.class_maps = class_maps
        if self.class_maps is None:
            self.class_maps = {column: self.class_to_idx(self.data[column]) for column in self.label_columns}
        self.tokenizer_args = TOKENIZER_ARGS
        if isinstance(tokenizer_args, dict):
            self.tokenizer_args.update(tokenizer_args)

    def class_to_idx(self, instance_labels):
        classes = sorted(set(instance_labels))
        return bidict({j: i for i, j in enumerate(classes)})

    def __len__(self):
        return self.data.num_rows

    def __getitem__(self, index):
        if isinstance(index, torch.Tensor) and index.dim() == 0:
            index = index.unsqueeze(0)
        selected = self.data[index]
        inputs = selected['text']
        if isinstance(inputs, list):
            inputs = inputs[0]
        inputs = preprocess_text(inputs)
        inputs = self.preprocessor(inputs, **self.tokenizer_args)
        labels_mapped = []
        for column in self.label_columns:
             val = selected[column]
             if isinstance(val, list):
                 val = val[0]
             labels_mapped.append(self.class_maps[column][val])
        # labels_mapped = [self.class_maps[column][selected[column]] for column in self.label_columns]
        inputs['labels'] = torch.tensor(np.array([labels_mapped]))
        return inputs


class MultitaskDatasetLong(Dataset):

    def __init__(self, dataset_dir: DatasetDict, preprocessor: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
                 split, class_maps=None, on_missing_class='warn', tokenizer_args={}):
        super().__init__()
        self.class_maps = class_maps
        self.load_dataset_slice(dataset_dir, split)
        self.tasks = list(self.class_maps.keys())
        self.preprocessor = preprocessor
        self.on_missing_class = on_missing_class
        self.tokenizer_args = TOKENIZER_ARGS
        if isinstance(tokenizer_args, dict):
            self.tokenizer_args.update(tokenizer_args)

    def load_dataset_slice(self, dataset_dir, split):
        self.task_splits = dict()
        self.records_index = list()
        class_maps = dict()

        def add_task_dataset(task_dir):
            dataset_split = load_task_dataset(task_dir, split)
            task_name = dataset_split[0]['task_name']
            self.task_splits[task_dir] = dataset_split
            class_maps[task_name] = self.class_to_idx(dataset_split['value'])
            return [(i, task_dir) for i in range(dataset_split.num_rows)]

        task_dirs = os.listdir(dataset_dir)
        self.records_index = [add_task_dataset(os.path.join(dataset_dir, task_dir)) for task_dir in task_dirs]
        self.records_index = list(itertools.chain.from_iterable(self.records_index))
        if self.class_maps is None:
            self.class_maps = class_maps

    @staticmethod
    def class_to_idx(instance_labels):
        classes = sorted(set(instance_labels))
        return bidict({j: i for i, j in enumerate(classes)})

    def __len__(self):
        return len(self.records_index)

    def __getitem__(self, index):
        selected_index = self.records_index[index]
        selected_entry = self.task_splits[selected_index[1]][selected_index[0]]
        inputs = selected_entry['text']
        inputs = preprocess_text(inputs)
        task = selected_entry['task_name']
        class_map = self.class_maps.get(task)
        if class_map is None:
            err_msg = f'Class map does not contain mappings for task {task}'
            if self.on_missing_class == "error":
                raise ValueError(err_msg)
            elif self.on_missing_class == 'warn':
                print(err_msg + ". Setting task value to None")
            elif self.on_missing_class == 'ignore':
                pass
        inputs = self.preprocessor(inputs, **self.tokenizer_args)
        inputs['tasks'] = self.tasks.index(task) if task in self.tasks else np.nan
        if class_map is not None:
            inputs['labels'] = class_map[selected_entry['value']]
        else:
            inputs['labels'] = np.nan  # Should be handled separately
        return inputs


if __name__ == '__main__':
    from transformers import AutoTokenizer
    model_path = "google-bert/bert-base-uncased"
    cache_dir='../cache'
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
    # multi_dataset = MultitaskDatasetLong("hf_berkeley_multi", tokenizer, split='train')
    multi_dataset = MultitaskDatasetWide("hf_berkeley_multi_wide", tokenizer, split='train')
    print(next(iter(multi_dataset)))
