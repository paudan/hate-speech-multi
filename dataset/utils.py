import numpy as np
from datasets import load_dataset, DatasetDict, concatenate_datasets
from sklearn.utils import compute_class_weight

def split_data_file(data_file, class_col=None, train_size=0.7):
    dataset: DatasetDict = load_dataset("parquet", data_files=[data_file], cache_dir='cache')
    train_split = dataset['train']
    if class_col:
        train_split = train_split.class_encode_column(class_col)
    splits = train_split.train_test_split(train_size=train_size, stratify_by_column=class_col)
    splits_val = splits['test'].train_test_split(train_size=0.5, stratify_by_column=class_col)
    return DatasetDict({
        'train': splits['train'],
        'validation': splits_val['train'],
        'test': splits_val['test']
    })

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