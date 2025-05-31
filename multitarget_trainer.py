# -*- coding: utf-8 -*-

import argparse
import json
import os
import pickle
from functools import partial
import numpy as np
import pandas as pd
import mlflow
import torch
from torch.utils.data import Subset
from transformers import AutoTokenizer, TrainingArguments, Trainer
from transformers.integrations import MLflowCallback
from transformers.trainer_callback import EarlyStoppingCallback
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score, 
    cohen_kappa_score, classification_report
)
from dataset.multitask_dataset import MultitaskDatasetWide
from dataset.utils import calculate_class_weights
from models.multitask_classifier import TransformerMultiTargetClassifier
from models.utils import create_trained_model

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

def compute_metrics(eval_preds, pos_label=1):
    all_logits, all_actual = eval_preds
    all_actual = np.squeeze(all_actual)
    prec_all = []
    rec_all = []
    f1_all = []
    roc_all = []
    acc_all = []
    for i in range(len(all_logits)):
        logits = all_logits[i]
        actual = all_actual[:, i]
        predictions = np.argmax(logits, axis=1)
        average_ = 'binary' if logits.shape[1] == 2 else 'micro'
        precision, recall, f1, _ = precision_recall_fscore_support(actual, predictions, average=average_, zero_division=0, pos_label=pos_label)
        try:
            roc_auc = roc_auc_score(actual, predictions, multi_class='ovr')
        except:
            roc_auc = np.nan
        accuracy = accuracy_score(actual, predictions)
        prec_all.append(precision)
        rec_all.append(recall)
        f1_all.append(f1)
        roc_all.append(roc_auc)
        acc_all.append(accuracy)
    return {
       'accuracy': np.nanmean(acc_all),
       'f1_score': np.nanmean(f1_all),
       'precision': np.nanmean(prec_all),
       'recall': np.nanmean(rec_all),
       'roc_auc': np.nanmean(roc_all)
    }


def calculate_scores(actual, predictions, task_name, average='binary', pos_label=1):
    precision, recall, f1, _ = precision_recall_fscore_support(actual, predictions, average=average, zero_division=0, pos_label=pos_label)
    try:
        roc_auc = roc_auc_score(actual, predictions, multi_class='ovr')
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


def train_eval_model(model_path, data_dir, cache_dir=None, output_dir='test-classifier', 
                     save_final=True, save_model_dir='final_classifier', 
                     batch_size=16, eval_batch_size=64, num_epochs=20, 
                     tuned_layers_count=0, dropout=0.1, pos_label=1, 
                     use_lora=False, use_corda=False, model_args={}, lora_args={}):
    set_seed()
    if use_corda is True:
        use_lora = True
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
    # To get complete mappings, complete dataset must be initialized
    class_maps = MultitaskDatasetWide(data_dir, tokenizer, load_all_data=True).class_maps
    train_dataset = MultitaskDatasetWide(data_dir, tokenizer, split='train', class_maps=class_maps)
    # class_maps = train_dataset.class_maps
    valid_dataset = MultitaskDatasetWide(data_dir, tokenizer, split='validation', class_maps=class_maps)
    test_dataset = MultitaskDatasetWide(data_dir, tokenizer, split='test', class_maps=class_maps)
    # train_dataset = Subset(train_dataset, torch.randint(low=0, high=5000, size=(1000,)))
    # valid_dataset = Subset(valid_dataset, torch.randint(low=0, high=5000, size=(500,)))
    # test_dataset = Subset(test_dataset, torch.randint(low=0, high=5000, size=(500,)))
    margs = dict(
        class_maps=class_maps,
        class_weights=calculate_class_weights(data_dir),
        tuned_layers_count=tuned_layers_count if not use_lora else 0,
        dropout=dropout
    )
    margs.update(model_args)
    trainer = Trainer(
        model=create_trained_model(
            TransformerMultiTargetClassifier, 
            model_path, 
            cache_dir=cache_dir,
            train_dataset=train_dataset,
            use_lora=use_lora,
            use_corda=use_corda,
            model_args=margs,
            lora_args=lora_args
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
        num_tasks = len(predictions.predictions)
        if isinstance(train_dataset, Subset):
            task_names = train_dataset.dataset.label_columns
        else:
            task_names = train_dataset.label_columns
        all_actual = np.squeeze(predictions.label_ids)
        for i in range(num_tasks):
            predicted = np.argmax(predictions.predictions[i], axis=1)
            average_ = 'binary' if predictions.predictions[i].shape[1] == 2 else 'micro'
            results.append(calculate_scores(all_actual[:, i], predicted, task_names[i], average=average_, pos_label=pos_label))
            print(task_names[i])
            print(classification_report(all_actual[:, i], predicted))
        results = pd.DataFrame(results)
        mlflow.log_dict(results.to_dict(), f"{trainer.model.__class__.__name__}.json")
        mlflow.log_dict(trainer.model.class_maps, 'class_maps.json')
    if save_final:
        os.makedirs(save_model_dir, exist_ok=True)
        with open(os.path.join(save_model_dir, 'class_maps.pkl'), 'wb') as f:
            pickle.dump(trainer.model.class_maps, f)
        tokenizer.save_pretrained(save_model_dir)
        trainer.model.save_pretrained(save_model_dir)
    return trainer.model, results


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate multitarget classifier")
    parser.add_argument('--model-path', type=str, help='Path to pre-trained model', required=True)
    parser.add_argument('--data-dir', type=str, help='Data directory', required=True)
    parser.add_argument('--cache-dir', type=str, default="cache", help='Cache directory')
    parser.add_argument('--output-dir', type=str, help='Output directory for training arguments', required=True)
    parser.add_argument('--save-final', action='store_true', default=True, help='Whether to save the final model')
    parser.add_argument('--model-dir', type=str, help='Directory to save model')
    parser.add_argument('--batch-size', type=int, default=16, help='Training batch size')
    parser.add_argument('--eval-batch-size', type=int, default=64, help='Evaluation batch size')
    parser.add_argument('--num-epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--tuned-layers-count', type=int, default=0, help='Number of tuned layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--pos-label', type=int, default=1, help='Positive label')
    parser.add_argument('--use-lora', action='store_true', default=False, help='Use LoRA')
    parser.add_argument('--use-corda', action='store_true', default=False, help='Use CORDA')
    parser.add_argument('--model-args', help='Additional model args', type=json.loads, required=False, default={})
    parser.add_argument('--lora-args', help='LoRa args', type=json.loads, required=False, default={})

    args = parser.parse_args()
    train_eval_model(
        model_path=args.model_path,
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        save_final=args.save_final,
        save_model_dir=args.model_dir,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_epochs=args.num_epochs,
        tuned_layers_count=args.tuned_layers_count,
        dropout=args.dropout,
        pos_label=args.pos_label,
        use_lora=args.use_lora,
        use_corda=args.use_corda,
        model_args=args.model_args,
        lora_args=args.lora_args
    )

if __name__ == "__main__":
    main()