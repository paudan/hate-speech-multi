# -*- coding: utf-8 -*-

__author__ = "Paulius Danėnas"
__copyright__ = "Copyright 2025, Paulius Danėnas, Kaunas University of Technology"
__maintainer__ = "Paulius Danėnas"
__email__ = "danpaulius@gmail.com"


import copy
from peft import LoraConfig, get_peft_model, TaskType
from peft.tuners.lora.corda import preprocess_corda
from tqdm import tqdm
from transformers import AutoConfig
import torch


def create_trained_model(model_class, model_path, cache_dir=None,
                         use_lora=False, use_corda=False, train_dataset=None,
                         model_args={}, lora_args={}):
    if use_corda is True and train_dataset is None:
        raise Exception('train_dataset must be set of CoRDA fine-tuning is used')
    if use_corda is True:
        use_lora = True
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu' 
    model = model_class.from_pretrained(
        model_path,
        config=AutoConfig.from_pretrained(model_path, cache_dir=cache_dir),
        cache_dir=cache_dir,
        device_map=device,
        **model_args
    )

    @torch.no_grad()    
    def run_model():
        model.eval()
        for batch in tqdm(train_dataset):
            # Force inference mode
            entry = copy.deepcopy(batch)
            if 'labels' in entry:
                entry['labels'] = None
            entry = entry.to(device)
            model(**entry)
        
    if use_lora:
        default_lora_args = {
            "task_type": TaskType.SEQ_CLS,
            "modules_to_save": ["classifier"]
        }
        lora_args.update(default_lora_args)
        config = LoraConfig(**lora_args)
        # model.add_adapter(config)
        if use_corda:
            preprocess_corda(model, config, run_model=run_model)
        model = get_peft_model(model, config)
    return model