# -*- coding: utf-8 -*-

import os
from functools import partial
import numpy as np
import mlflow
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoConfig, TrainingArguments, Trainer
from transformers.integrations import MLflowCallback
from transformers.trainer_callback import EarlyStoppingCallback
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, cohen_kappa_score
from models.multitask_classifier import SimpleTransformerClassifier
from dataset.multitask_dataset import SimpleDataset

os.environ["TOKENIZERS_PARALLELISM"] = 'true'
os.environ['WANDB_DISABLED'] = 'true'
SEED = 42


def set_seed():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

def get_training_args(output_dir, batch_size=8, num_epochs=20, eval_batch_size=64):
    return TrainingArguments(
        output_dir=output_dir,
        metric_for_best_model='eval_accuracy',
        load_best_model_at_end=True,
        greater_is_better=True,
        eval_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        logging_strategy='epoch',
        log_level='info',
        logging_first_step=True,
        save_strategy='epoch',
        save_total_limit=2,
        num_train_epochs=num_epochs,
        auto_find_batch_size=False,
        ignore_data_skip=True,
        disable_tqdm=False,
        overwrite_output_dir=True,
        # lr_scheduler_type="cosine",
        fp16_full_eval=False,
        fp16=False,
        fp16_opt_level='O1',
        report_to=['tensorboard'],
        seed=SEED,
        data_seed=SEED
    )

def compute_metrics(eval_preds, pos_label=0):
    logits, actual = eval_preds
    predictions = np.argmax(logits, axis=-1)
    average_ = 'binary' if logits.shape[1] == 2 else 'micro'
    precision, recall, f1, _ = precision_recall_fscore_support(actual, predictions, average=average_, zero_division=0, pos_label=pos_label)
    try:
        roc_auc = roc_auc_score(actual, predictions)
    except:
        roc_auc = np.nan
    return {
       'accuracy': accuracy_score(actual, predictions),
       'f1_score': f1,
       'precision': precision,
       'recall': recall,
       'roc_auc': roc_auc
    }

def input_generator(texts, targets):
    for txt, label in zip(texts, targets):
        yield {"text": txt, "labels": label}

def calculate_scores(actual, predictions, task_name, average='binary', pos_label=0):
    precision, recall, f1, _ = precision_recall_fscore_support(actual, predictions, average=average, zero_division=0, pos_label=pos_label)
    try:
        roc_auc = roc_auc_score(actual, predictions)
    except:
        roc_auc = np.nan
    return {
       'task': task_name,
       'accuracy': accuracy_score(actual, predictions),
       'f1_score': f1,
       'precision': precision,
       'recall': recall,
       'kappa': cohen_kappa_score(actual, predictions),
       'roc_auc': roc_auc
    }  


def train_eval_model(model_path, inputs, targets, train_size=0.7, valid_size=0.15,
                     cache_dir=None, output_dir='test-classifier', task_name=None,
                     save_final=True, save_model_dir='final_classifier', 
                     batch_size=16, eval_batch_size=64, num_epochs=20, 
                     tuned_layers_count=0, dropout=0.1, pos_label=0, **model_args):
    set_seed()
    targets = list(map(int, targets))
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
    tokenize_fn = lambda examples: tokenizer(examples["text"], padding="max_length", truncation=True)
    train_dst = Dataset.from_generator(lambda: input_generator(inputs, targets)).train_test_split(train_size=train_size, shuffle=True, seed=SEED)
    tokenized_dataset = train_dst.map(tokenize_fn, batched=True)
    train_dataset = tokenized_dataset["train"].shuffle(seed=SEED)
    test_splits = tokenized_dataset["test"].train_test_split(train_size=valid_size/(1-train_size))
    valid_dataset = test_splits["train"].shuffle(seed=SEED)
    test_dataset = test_splits["test"].shuffle(seed=SEED)
    trainer = Trainer(
        model=SimpleTransformerClassifier.from_pretrained(
            model_path,
            config=AutoConfig.from_pretrained(model_path, cache_dir=cache_dir),
            cache_dir=cache_dir,
            device_map='cuda' if torch.cuda.is_available() else 'cpu',
            num_labels=len(set(targets)),
            tuned_layers_count=tuned_layers_count,
            dropout=dropout
        ),
        args=get_training_args(output_dir, batch_size, num_epochs, eval_batch_size),
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=partial(compute_metrics, pos_label=pos_label),
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=5),
            MLflowCallback()
        ]
    )
    with mlflow.start_run():
        trainer.train()
        predictions = trainer.predict(test_dataset)
        results = []
        predicted = np.argmax(predictions.predictions, axis=1)
        average_ = 'binary' if predictions.predictions.shape[1] == 2 else 'micro'
        results = calculate_scores(predictions.label_ids, predicted, task_name, average=average_, pos_label=pos_label)
        mlflow.log_dict(results, f"{trainer.model.__class__.__name__}.json")
    if save_final:
        os.makedirs(save_model_dir, exist_ok=True)
        tokenizer.save_pretrained(save_model_dir)
        trainer.model.save_pretrained(save_model_dir)
    return trainer.model, results
