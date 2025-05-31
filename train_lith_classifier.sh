#!/bin/bash

BATCH_SIZE=64
EVAL_BATCH_SIZE=128
NUM_EPOCHS=20
BASE_MODEL="intfloat/multilingual-e5-large"
CACHE_DIR="../cache"

python3 multitarget_trainer.py \
  --model-path $BASE_MODEL \
  --cache-dir $CACHE_DIR \
  --data-dir "data/lith_dataset_multi_wide" \
  --model-dir "outputs/lith_classifier_multi_wide" \
  --batch-size $BATCH_SIZE \
  --eval-batch-size $EVAL_BATCH_SIZE \
  --num-epochs $NUM_EPOCHS \
  --tuned-layers-count 0 \
  --dropout 0.1 \
  --output-dir "test-classifier-lith" \
| tee results/lith-dataset-multitarget-nofinetune.txt

python3 multitarget_trainer.py \
  --model-path $BASE_MODEL \
  --cache-dir $CACHE_DIR \
  --data-dir "data/lith_dataset_multi_wide" \
  --model-dir "outputs/lith_classifier_multi_wide" \
  --batch-size $BATCH_SIZE \
  --eval-batch-size $EVAL_BATCH_SIZE \
  --num-epochs $NUM_EPOCHS \
  --tuned-layers-count -1 \
  --dropout 0.1 \
  --output-dir "test-classifier-lith" \
| tee results/lith-dataset-multitarget-finetune.txt

python3 multitarget_trainer.py \
  --model-path $BASE_MODEL \
  --cache-dir $CACHE_DIR \
  --data-dir "data/lith_dataset_multi_wide" \
  --model-dir "outputs/lith_classifier_multi_wide_lora" \
  --batch-size $BATCH_SIZE \
  --eval-batch-size $EVAL_BATCH_SIZE \
  --num-epochs $NUM_EPOCHS \
  --tuned-layers-count 0 \
  --dropout 0.1 \
  --output-dir "test-classifier-lith-lora" \
  --use-lora \
  --lora-args '{
        "init_lora_weights": "olora",
        "target_modules": "all-linear"
    }' \
| tee results/lith-dataset-multitarget-lora.txt


python3 multitask_trainer.py \
  --model-path $BASE_MODEL \
  --cache-dir $CACHE_DIR \
  --data-dir "data/lith_dataset_multi" \
  --model-dir "outputs/lith_classifier_multi" \
  --batch-size $BATCH_SIZE \
  --eval-batch-size $EVAL_BATCH_SIZE \
  --num-epochs $NUM_EPOCHS \
  --tuned-layers-count 0 \
  --dropout 0.1 \
  --output-dir "test-classifier-lith" \
| tee results/lith-dataset-multitask-nofinetune.txt

python3 multitask_trainer.py \
  --model-path $BASE_MODEL \
  --cache-dir $CACHE_DIR \
  --data-dir "data/lith_dataset_multi" \
  --model-dir "outputs/lith_classifier_multi" \
  --batch-size $BATCH_SIZE \
  --eval-batch-size $EVAL_BATCH_SIZE \
  --num-epochs $NUM_EPOCHS \
  --tuned-layers-count -1 \
  --dropout 0.1 \
  --output-dir "test-classifier-lith" \
| tee results/lith-dataset-multitask-finetune.txt

python3 multitask_trainer.py \
  --model-path $BASE_MODEL \
  --cache-dir $CACHE_DIR \
  --data-dir "data/lith_dataset_multi" \
  --model-dir "outputs/lith_classifier_multi_lora" \
  --batch-size $BATCH_SIZE \
  --eval-batch-size $EVAL_BATCH_SIZE \
  --num-epochs $NUM_EPOCHS \
  --tuned-layers-count 0 \
  --dropout 0.1 \
  --output-dir "test-classifier-lith-lora" \
  --use-lora \
  --lora-args '{
        "init_lora_weights": "olora",
        "target_modules": "all-linear"
    }' \
| tee results/lith-dataset-multitask-lora.txt
