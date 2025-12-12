# -*- coding: utf-8 -*-

__author__ = "Paulius Danėnas"
__copyright__ = "Copyright 2025, Paulius Danėnas, Kaunas University of Technology"
__maintainer__ = "Paulius Danėnas"
__email__ = "danpaulius@gmail.com"


from collections import OrderedDict
from typing import Optional
import torch
import torch.nn as nn
from transformers import ModernBertConfig, ModernBertModel
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.modernbert.modeling_modernbert import ModernBertPredictionHead


class ModernBertBackbone(ModernBertModel):
    config_class=ModernBertConfig

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
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        sliding_window_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        indices: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        batch_size: Optional[int] = None,
        seq_len: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        outputs = super().forward(
            input_ids,
            attention_mask=attention_mask,
            sliding_window_mask=sliding_window_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            indices=indices,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            batch_size=batch_size,
            seq_len=seq_len,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict   
        )
        x = outputs[0]
        return x


class SimpleModernBertClassifier(ModernBertBackbone):

    def __init__(self, config, num_labels=2, dropout=0.1, tuned_layers_count=0, pooling_type=None):
        super().__init__(config, tuned_layers_count=tuned_layers_count)
        self.pooling = pooling_type or self.config.classifier_pooling
        if self.pooling not in ('cls', 'mean'):
            self.pooling = self.config.classifier_pooling
        self.num_labels = num_labels
        self.head = ModernBertPredictionHead(config)
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

    def forward(self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        sliding_window_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        indices: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        batch_size: Optional[int] = None,
        seq_len: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None
    ):
        x = self.embed(
            input_ids,
            attention_mask=attention_mask,
            sliding_window_mask=sliding_window_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            indices=indices,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            batch_size=batch_size,
            seq_len=seq_len,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict            
        )
        if self.pooling == "cls":
            x = x[:, 0]
        elif self.pooling == "mean":
            x = (x * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
            
        x = self.head(x)
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
    

class ModernBertMultiHeadClassifier(ModernBertBackbone):

    def __init__(self, config, class_maps, dropout=0.1, tuned_layers_count=0, class_weights=None, pooling_type=None):
        super().__init__(config, tuned_layers_count=tuned_layers_count)
        self.class_maps = class_maps
        self.class_weights = class_weights
        self.pooling = pooling_type or self.config.classifier_pooling
        if self.pooling not in ('cls', 'mean'):
            self.pooling = self.config.classifier_pooling
        self.columns = list(self.class_maps.keys())
        self.dropout = dropout
        self.heads = nn.ModuleList(list(map(self._create_head, self.columns)))

    def _create_head(self, column):
        num_labels = len(self.class_maps.get(column))
        layers = [
            ModernBertPredictionHead(self.config),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.config.hidden_size, num_labels)
        ]
        return nn.ModuleList(layers)

    def _fix_attention_mask(self, 
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        batch_size: Optional[int] = None,
        seq_len: Optional[int] = None,                       
    ):
        self._maybe_set_compile()
        if batch_size is None and seq_len is None:
            if inputs_embeds is not None:
                batch_size, seq_len = inputs_embeds.shape[:2]
            else:
                batch_size, seq_len = input_ids.shape[:2]
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_len), device=device, dtype=torch.bool)
        return batch_size, seq_len, attention_mask 


class ModernBertMultiTargetClassifier(ModernBertMultiHeadClassifier):

    def forward(self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        sliding_window_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        indices: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        batch_size: Optional[int] = None,
        seq_len: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None
    ):
        if input_ids is not None:
            input_ids = torch.squeeze(input_ids, dim=1)
        if attention_mask is not None:
            attention_mask = torch.squeeze(attention_mask, dim=1)
        if sliding_window_mask is not None:
            sliding_window_mask = torch.squeeze(sliding_window_mask, dim=1)
        if position_ids is not None:
            position_ids = torch.squeeze(position_ids, dim=1)
        if labels is not None:
            labels = torch.squeeze(labels, dim=1)
        batch_size, seq_len, attention_mask = self._fix_attention_mask(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            batch_size=batch_size,
            seq_len=seq_len
        )
        x = self.embed(
            input_ids,
            attention_mask=attention_mask,
            sliding_window_mask=sliding_window_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            indices=indices,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            batch_size=batch_size,
            seq_len=seq_len,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict            
        )
        if self.pooling == "cls":
            x = x[:, 0]
        elif self.pooling == "mean":
            x = (x * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        
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
    

class ModernBertMultiTaskClassifier(ModernBertMultiHeadClassifier):

    def forward(self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        sliding_window_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        indices: Optional[torch.Tensor] = None,
        tasks: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        batch_size: Optional[int] = None,
        seq_len: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None
    ):
        if input_ids is not None:
            input_ids = torch.squeeze(input_ids, dim=1)
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
        if attention_mask is not None:
            attention_mask = torch.squeeze(attention_mask, dim=1)
        if sliding_window_mask is not None:
            sliding_window_mask = torch.squeeze(sliding_window_mask, dim=1)
        if position_ids is not None:
            position_ids = torch.squeeze(position_ids, dim=1)
        if isinstance(labels, int):
            labels = torch.tensor([labels], dtype=torch.long)
        if isinstance(tasks, int):
            tasks = torch.tensor([tasks], dtype=torch.long)
        batch_size, seq_len, attention_mask = self._fix_attention_mask(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            batch_size=batch_size,
            seq_len=seq_len
        )
        x = self.embed(
            input_ids,
            attention_mask=attention_mask,
            sliding_window_mask=sliding_window_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            indices=indices,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            batch_size=batch_size,
            seq_len=seq_len,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=False          
        )
        if self.pooling == "cls":
            x = x[:, 0]
        elif self.pooling == "mean":
            x = (x * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)

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
