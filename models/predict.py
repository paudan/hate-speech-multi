# -*- coding: utf-8 -*-

__author__ = "Paulius Danėnas"
__copyright__ = "Copyright 2025, Paulius Danėnas, Kaunas University of Technology"
__maintainer__ = "Paulius Danėnas"
__email__ = "danpaulius@gmail.com"


from abc import abstractmethod
import os
import pickle
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BatchEncoding
import seaborn as sns
import matplotlib.pyplot as plt
from captum.attr import visualization as viz
from captum.attr import LayerConductance, LayerIntegratedGradients
from .multitask_bert import BertMultiTargetClassifier, BertMultiTaskClassifier


class BaseMultiTaskModel:

    def __init__(self, base_class, model_dir, ig_batch_size=2, cache_dir=None):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        with open(os.path.join(model_dir, 'class_maps.pkl'), 'rb') as f:
            class_maps = pickle.load(f)
        self.class_maps = class_maps
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, cache_dir=cache_dir)
        self.model = base_class.from_pretrained(model_dir, cache_dir=cache_dir, class_maps=class_maps).to(self.device)
        self.ig_batch_size = ig_batch_size
        self.model.eval()
        self.model.zero_grad()

    def get_processed(self, text):
        return self.tokenizer.encode_plus(text, 
            add_special_tokens=True, 
            # padding="max_length", 
            return_attention_mask=True, 
            return_token_type_ids=True,
            return_tensors="pt"
        ).to(self.device)

    @abstractmethod
    def predict(self, text):
        raise Exception("Not implemented")

    def _forward_func(self, input_ids, task=None, token_type_ids=None, attention_mask=None):
        prediction = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        prediction = {task: pred for task, pred in prediction.items()}
        return prediction[task].max(1).values

    def _forward_func2(self, inputs_embeds, task=None, attention_mask=None):
        prediction = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        prediction = {task: pred for task, pred in prediction.items()}
        return prediction[task].max(1).values
    
    def get_attributions(self, text, task, n_steps=50):
        processed = self.get_processed(text)

        def summarize_attributions(attributions):
            attributions = attributions.sum(dim=-1).squeeze(0)
            attributions = attributions / torch.norm(attributions)
            return attributions

        lig = LayerIntegratedGradients(self._forward_func, self.model.base_model.embeddings)
        attributions, delta_start = lig.attribute(inputs=processed['input_ids'], 
            additional_forward_args=(task, processed['token_type_ids'], processed['attention_mask']),
            internal_batch_size=self.ig_batch_size,
            return_convergence_delta=True,
            n_steps=n_steps
        )
        attributions = summarize_attributions(attributions)
        layer_attrs = []
        input_embeddings = self.model.base_model.embeddings(
            input_ids=processed['input_ids'], 
            token_type_ids=processed['token_type_ids']
        )
        for i in range(self.model.config.num_hidden_layers):
            lc = LayerConductance(self._forward_func2, self.model.base_model.encoder.layer[i])
            layer_attributions = lc.attribute(
                inputs=input_embeddings, 
                additional_forward_args=(task, processed['attention_mask']),
                internal_batch_size=self.ig_batch_size,
                n_steps=n_steps
            )
            layer_attrs.append(summarize_attributions(layer_attributions).cpu().detach().tolist())
        return attributions, layer_attrs, delta_start
    
    def get_predicted_probs(self, text, task):
        processed = self.get_processed(text)
        probs = self.model(**processed)
        return probs[task]
    
    def agg_attributions(self, tokens, attributions, layer_attrs):
        if not isinstance(layer_attrs, torch.Tensor):
            layer_attrs = torch.tensor(layer_attrs)
        splits = []
        current_left = current_right = 0
        for i, tok in enumerate(tokens):
            if tok[0] != '#' and tok[-1] != '#':
                if i < len(tokens)-1 and tokens[i+1][0] != '#':
                    splits.append((current_left, current_left))
                    current_left = i + 1
                elif i == len(tokens)-1:
                    splits.append((i, i))
                    break
                else:
                    current_right = i + 1
            elif tok[0] == '#' and tok[-1] != '#':
                if i < len(tokens)-1 and tokens[i+1][0] != '#':
                    splits.append((current_left, current_right))
                    current_left = i + 1
                else:
                    current_right = i + 1
        detokenized = [''.join(map(lambda x: x.strip('#'), tokens[left:right+1])) for left, right in splits]
        attributions = torch.hstack([torch.max(attributions[left:right+1], dim=-1, keepdim=True)[0] for left, right in splits]).squeeze()
        layer_attrs = torch.hstack([torch.max(layer_attrs[:, left:right+1], dim=-1, keepdim=True)[0] for left, right in splits]).squeeze()
        return detokenized, attributions, layer_attrs

    def plot_interpretability(self, text, task, n_steps=50, detokenize=False):
        attributions, layer_attrs, delta_start = self.get_attributions(text, task, n_steps)
        probs = self.get_predicted_probs(text, task)
        all_tokens = self.tokenizer.tokenize(text, add_special_tokens=True)
        if detokenize:
            all_tokens, attributions, layer_attrs = self.agg_attributions(all_tokens, attributions, layer_attrs)
#        print(attributions, probs, all_tokens)
        position_vis = viz.VisualizationDataRecord(
            attributions,
            torch.max(probs, dim=1).values[0].detach().cpu().numpy(),
            torch.argmax(probs[0], keepdim=True).cpu().numpy(),
            torch.argmax(probs[0], keepdim=True).cpu().numpy(),
            "NA",
            attributions.sum(),       
            all_tokens,
            delta_start
        )
        task_name = list(self.class_maps.keys())[task]
        print(f"Visualization for target {task_name}")
        viz.visualize_text([position_vis])
        fig, ax = plt.subplots(figsize=(15,5))
        xticklabels=all_tokens
        yticklabels=list(range(1,13))
        ax = sns.heatmap(np.array(layer_attrs), xticklabels=xticklabels, yticklabels=yticklabels, linewidth=0.2)
        plt.xlabel('Tokens')
        plt.ylabel('Layers')
        plt.show()
        return plt


class MultiTargetModel(BaseMultiTaskModel):

    def __init__(self, model_dir, model_class=BertMultiTargetClassifier, **args):
        super().__init__(model_class, model_dir, **args)

    def predict(self, text):
        processed = self.get_processed(text)
        results = self.model(**processed)
        results = {task: pred.detach().cpu().numpy().squeeze().tolist() for task, pred in results.items()}
        return results
    

class MultiTaskModel(BaseMultiTaskModel):

    def __init__(self, model_dir, model_class=BertMultiTaskClassifier, **args):
        super().__init__(model_class, model_dir, **args)

    def predict(self, inputs, task=None):
        processed = None
        tasks = list(self.class_maps.keys())
        if isinstance(inputs, str):
            if task is None:
                tokenized = [inputs] * len(tasks)
                task_labels = torch.arange(len(tasks))
            else:
                tokenized = inputs
                task_labels = task
            processed = self.tokenizer(tokenized, 
                add_special_tokens=True, 
                # truncation=True, 
                padding="max_length", 
                return_attention_mask=True, 
                return_token_type_ids=True,
                return_tensors="pt"
            )
            processed.update({'tasks': task_labels})
        elif isinstance(inputs, (dict, BatchEncoding)):
            processed = inputs
        else:
            raise Exception("Invalid inputs format")
        processed = processed.to(self.device)
        results = self.model(**processed)
        if task is None:
            results = {tasks[i]: pred.detach().cpu().numpy() for i, pred in enumerate(results[0])}
        else:
            pred, task = results
            task_ind = task.detach().cpu().numpy()[0]
            results = {tasks[task_ind]: pred[0].detach().cpu().numpy()}
        return results
    
    def _forward_func(self, input_ids, task=None, token_type_ids=None, attention_mask=None):
        prediction = self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, tasks=task)
        return prediction[0][0].unsqueeze(0).max(1).values

    def _forward_func2(self, inputs_embeds, task=None, attention_mask=None):
        prediction = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, tasks=task)
        return prediction[0][0].unsqueeze(0).max(1).values
    
    def get_predicted_probs(self, text, task):
        processed = self.tokenizer(text, 
            add_special_tokens=True, 
            # truncation=True, 
            padding="max_length", 
            return_attention_mask=True, 
            return_token_type_ids=True,
            return_tensors="pt"
        )
        processed.update({'tasks': task})
        processed = processed.to(self.device)
        results = self.model(**processed)
        return results[0][0].unsqueeze(0)
    
    def evaluate(self, dataset):
        predictions, labels = [], []
        loader = DataLoader(dataset, batch_size=1)
        for inputs in tqdm(loader):
            if np.isnan(inputs['tasks']):
                continue
            labels.append(inputs['labels'].detach().cpu().numpy())
            del inputs['labels']
            predicted = self.predict(inputs, task=inputs['tasks'])
            predictions.append(predicted)
        return [{
                'task': list(pred.items())[0][0], 
                'predicted': np.argmax(list(pred.items())[0][1]), 
                'actual': label[0]
            } for pred, label in zip(predictions, labels)
        ]
