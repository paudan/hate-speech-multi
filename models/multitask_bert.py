# -*- coding: utf-8 -*-

__author__ = "Paulius Danėnas"
__copyright__ = "Copyright 2025, Paulius Danėnas, Kaunas University of Technology"
__maintainer__ = "Paulius Danėnas"
__email__ = "danpaulius@gmail.com"


from collections import OrderedDict
from typing import Optional
import torch
import torch.nn as nn
from transformers import BertConfig, BertModel
from transformers.modeling_outputs import SequenceClassifierOutput


class BertBackbone(BertModel):
    config_class=BertConfig

    def __init__(self, config, tuned_layers_count=0):
        super().__init__(config)
        # tuned_layers_count = -1 means fine-tune full model
        if tuned_layers_count > -1:
            for param in self.base_model.parameters():
                param.requires_grad = False
            # Unfreeze last frozen_layers_count layers for finetuning
            if tuned_layers_count > 0:
                for idx in range(-1, -tuned_layers_count-1, -1):
                    for param in self.encoder.layer[idx].parameters():
                        param.requires_grad = True

    def embed(self,
        input_ids: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        outputs = super().forward(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        x = outputs[1]
        return x


class SimpleBertClassifier(BertBackbone):

    def __init__(self, config, num_labels=2, dropout=0.1, use_layer_norm=False, tuned_layers_count=0):
        super().__init__(config, tuned_layers_count=tuned_layers_count)
        self.use_layer_norm = use_layer_norm
        self.num_labels = num_labels
        if use_layer_norm:
            self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

    def forward(self,
        input_ids: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,  
    ):
        x = self.embed(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict            
        )
        if self.use_layer_norm:
            x = self.layernorm(x)
        x = self.dropout(x)
        logits = self.classifier(x)
        if labels is None:
            return torch.softmax(logits, dim=-1)
        loss = None
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        if not return_dict:
            return ((loss,) + (logits,)) if loss is not None else logits
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits
        )        
    

class BertMultiHeadClassifier(BertBackbone):

    def __init__(self, config, class_maps, dropout=0.1, use_layer_norm=False, tuned_layers_count=0, class_weights=None):
        super().__init__(config, tuned_layers_count=tuned_layers_count)
        self.class_maps = class_maps
        self.class_weights = class_weights
        self.columns = list(self.class_maps.keys())
        self.dropout = dropout
        self.use_layer_norm = use_layer_norm
        self.heads = nn.ModuleList(list(map(self._create_head, self.columns)))

    def _create_head(self, column):
        num_labels = len(self.class_maps.get(column))
        layers = list()
        if self.use_layer_norm:
            layernorm = nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps)
            layers.append(layernorm)
        layers.extend([
            nn.Dropout(p=self.dropout),
            nn.Linear(self.config.hidden_size, num_labels)
        ])
        return nn.ModuleList(layers)


class BertMultiTargetClassifier(BertMultiHeadClassifier):

    def forward(self,
        input_ids: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        if input_ids is not None:
            input_ids = torch.squeeze(input_ids, dim=1)
        if token_type_ids is not None:
            token_type_ids = torch.squeeze(token_type_ids, dim=1)
        if attention_mask is not None:
            attention_mask = torch.squeeze(attention_mask, dim=1)
        if labels is not None:
            labels = torch.squeeze(labels, dim=1)
        x = self.embed(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict            
        )
        num_tasks = len(self.columns)
        total_loss = 0
        all_logits = list()
        all_preds = dict()
        for column_index in range(num_tasks):
            column = self.columns[column_index]
            num_labels = len(self.class_maps.get(column))
            col_weights = None
            if self.class_weights:
                col_weights = self.class_weights.get(column)
                if col_weights is not None:
                    col_weights = torch.tensor(col_weights, device=self.device, dtype=torch.float)
            out = x
            for module in self.heads[column_index]:
                out = module(out)
            logits = out
            if labels is None:
                all_preds[column] = torch.softmax(logits, dim=-1)
            else:
                loss = None
                loss_fct = nn.CrossEntropyLoss(weight=col_weights)
                loss = loss_fct(logits.view(-1, num_labels), labels[:, column_index].view(-1))
                total_loss += loss
                all_logits.append(logits)
        if labels is None:
            return all_preds
        if not return_dict:
            return (total_loss, all_logits) if loss is not None else all_logits
        return SequenceClassifierOutput(loss=total_loss, logits=all_logits)
    

class BertMultiTaskClassifier(BertMultiHeadClassifier):

    def forward(self,
        input_ids: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        tasks: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None
    ):
        if input_ids is not None:
            input_ids = torch.squeeze(input_ids, dim=1)
        if token_type_ids is not None:
            token_type_ids = torch.squeeze(token_type_ids, dim=1)
        if attention_mask is not None:
            attention_mask = torch.squeeze(attention_mask, dim=1)
        if isinstance(labels, int):
            labels = torch.tensor([labels], dtype=torch.long)
        if isinstance(tasks, int):
            tasks = torch.tensor([tasks], dtype=torch.long)
        x = self.embed(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=False            
        )
        # Required for integrated gradients: x dim[0] might be doubled, fix that with broadcast
        if tasks.shape[0] == 1 and x.shape[0] > tasks.shape[0]:
            tasks = torch.full((x.shape[0], ), tasks[0])
        if isinstance(tasks, torch.Tensor):
            batch_tasks = torch.unique(tasks)
        elif isinstance(tasks, int):
            batch_tasks = torch.tensor([tasks], dtype=torch.int)
        else:
            raise Exception(f"Invalid tasks type: {type(tasks)}")
        total_loss = 0
        all_logits = OrderedDict()
        all_preds = OrderedDict()
        for task in batch_tasks:
            task_index = tasks == task
            column = self.columns[task]
            num_labels = len(self.class_maps.get(column))
            col_weights = None
            if self.class_weights:
                col_weights = self.class_weights.get(column)
                if col_weights is not None:
                    col_weights = torch.tensor(col_weights, device=self.device, dtype=torch.float)
            out = x[task_index]
            for module in self.heads[task]:
                out = module(out)
            logits = out
            current_index = task_index.nonzero().squeeze(dim=1).cpu().numpy()
            if labels is None:
                all_preds.update(dict(zip(current_index, torch.softmax(logits, dim=-1))))
            else:
                loss = None
                loss_fct = nn.CrossEntropyLoss(weight=col_weights)
                loss = loss_fct(logits.view(-1, num_labels), labels[task_index].view(-1))
                total_loss += loss
                all_logits.update(dict(zip(current_index, logits)))
        all_logits = list(all_logits.values())
        all_preds = list(all_preds.values())
        if labels is None:
            return (all_preds, tasks)
        return (total_loss, all_logits, tasks)
