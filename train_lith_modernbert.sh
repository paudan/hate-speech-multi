#!/bin/bash

BATCH_SIZE=32
EVAL_BATCH_SIZE=128
NUM_EPOCHS=20
MAX_LENGTH=256
BASE_MODEL="VSSA-SDSA/LT-MLKM-modernBERT"
CACHE_DIR="cache"

python3 multitask_trainer.py \
  --model-type=modernbert \
  --model-path $BASE_MODEL \
  --cache-dir $CACHE_DIR \
  --data-dir "data/lith_dataset_multi" \
  --model-dir "outputs/lith_multi_mlkm_lora" \
  --batch-size $BATCH_SIZE \
  --eval-batch-size $EVAL_BATCH_SIZE \
  --num-epochs $NUM_EPOCHS \
  --tuned-layers-count 0 \
  --dropout 0.1 \
  --output-dir "output-lith-multi-mlkm-lora" \
  --use-lora \
  --lora-args '{
      "init_lora_weights": "olora",
      "target_modules": "all-linear"
    }' \
  --tokenizer-args '{
      "return_token_type_ids": "false",
      "max_length": 256
    }'    
| tee results/lith_multi_mlkm_lora.txt