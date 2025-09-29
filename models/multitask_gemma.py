# -*- coding: utf-8 -*-

__author__ = "Paulius Danėnas"
__copyright__ = "Copyright 2025, Paulius Danėnas, Kaunas University of Technology"
__maintainer__ = "Paulius Danėnas"
__email__ = "danpaulius@gmail.com"


from collections import OrderedDict
from typing import Optional
import torch
import torch.nn as nn
from transformers import Gemma3TextModel, Gemma3TextConfig
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from transformers.utils import logging
from transformers.utils.generic import TransformersKwargs
from transformers.cache_utils import Cache
from transformers.processing_utils import Unpack

logger = logging.get_logger(__name__)


class SimpleGemmaClassifier(Gemma3TextModel):
    config_class=Gemma3TextConfig

    def __init__(self, config, num_labels=2, dropout=0.1, use_layer_norm=False):
        super().__init__(config)
        self.use_layer_norm = use_layer_norm
        self.num_labels = num_labels
        if use_layer_norm:
            self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> SequenceClassifierOutputWithPast:
        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            token_type_ids=token_type_ids,
            use_cache=use_cache,
            **kwargs,
        )
        x = transformer_outputs.last_hidden_state
        if self.use_layer_norm:
            x = self.layernorm(x)
        x = self.dropout(x)
        logits = self.classifier(x)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            last_non_pad_token = -1
        elif input_ids is not None:
            # To handle both left- and right- padding, we take the rightmost token that is not equal to pad_token_id
            non_pad_mask = (input_ids != self.config.pad_token_id).to(logits.device, torch.int32)
            token_indices = torch.arange(input_ids.shape[-1], device=logits.device, dtype=torch.int32)
            last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)
        else:
            last_non_pad_token = -1
            logger.warning_once(
                f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
            )

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), last_non_pad_token]
        if labels is None:
            return torch.softmax(pooled_logits, dim=-1)
        loss = None
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )        
    

class GemmaMultiHeadClassifier(Gemma3TextModel):
    config_class=Gemma3TextConfig

    def __init__(self, config, class_maps, dropout=0.1, use_layer_norm=False, class_weights=None):
        super().__init__(config)
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
    
    def _calculate_pooled_logits(
        self,
        logits: torch.FloatTensor,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]
        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            last_non_pad_token = -1
        elif input_ids is not None:
            # To handle both left- and right- padding, we take the rightmost token that is not equal to pad_token_id
            non_pad_mask = (input_ids != self.config.pad_token_id).to(logits.device, torch.int32)
            token_indices = torch.arange(input_ids.shape[-1], device=logits.device, dtype=torch.int32)
            last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)
        else:
            last_non_pad_token = -1
            logger.warning_once(
                f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
            )
        x = logits[torch.arange(batch_size, device=logits.device), last_non_pad_token]
        return x


class GemmaMultiTargetClassifier(GemmaMultiHeadClassifier):

    def forward(self,
        input_ids: torch.LongTensor = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> SequenceClassifierOutputWithPast:
        if input_ids is not None:
            input_ids = torch.squeeze(input_ids, dim=1)
        if token_type_ids is not None:
            token_type_ids = torch.squeeze(token_type_ids, dim=1)
        if attention_mask is not None:
            attention_mask = torch.squeeze(attention_mask, dim=1)
        if labels is not None:
            labels = torch.squeeze(labels, dim=1)
        transformer_outputs = super().forward(
            input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            token_type_ids=token_type_ids,
            use_cache=use_cache,
            **kwargs,
        )
        x = transformer_outputs.last_hidden_state
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
            logits = self._calculate_pooled_logits(logits, input_ids, inputs_embeds)
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
        return SequenceClassifierOutputWithPast(loss=total_loss, logits=all_logits)
    

class GemmaMultiTaskClassifier(GemmaMultiHeadClassifier):

    def forward(self,
        input_ids: torch.LongTensor = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        tasks: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> SequenceClassifierOutputWithPast:
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
        transformer_outputs = super().forward(
            input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            token_type_ids=token_type_ids,
            use_cache=use_cache,
            **kwargs,
        )
        x = transformer_outputs.last_hidden_state
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
            pooled_logits = self._calculate_pooled_logits(
                logits, 
                input_ids = input_ids[task_index] if input_ids is not None else None,
                inputs_embeds = inputs_embeds[task_index] if inputs_embeds is not None else None
            )
            current_index = task_index.nonzero().squeeze(dim=1).cpu().numpy()
            if labels is None:
                all_preds.update(dict(zip(current_index, torch.softmax(pooled_logits, dim=-1))))
            else:
                loss = None
                loss_fct = nn.CrossEntropyLoss(weight=col_weights)
                loss = loss_fct(pooled_logits.view(-1, num_labels), labels[task_index].view(-1))
                total_loss += loss
                all_logits.update(dict(zip(current_index, pooled_logits)))
        all_logits = list(all_logits.values())
        all_preds = list(all_preds.values())
        if labels is None:
            return (all_preds, tasks)
        return (total_loss, all_logits, tasks)
