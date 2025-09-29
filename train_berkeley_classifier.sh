#!/bin/bash

BATCH_SIZE=64
EVAL_BATCH_SIZE=128
NUM_EPOCHS=20
BASE_MODEL="intfloat/e5-small-v2"
CACHE_DIR="cache"

python3 multitarget_trainer.py \
  --model-path $BASE_MODEL \
  --cache-dir $CACHE_DIR \
  --data-dir "data/hf_berkeley_multi_wide" \
  --output-dir "test-classifier-berkeley" \
  --model-dir "outputs/berkeley_classifier_multi_wide" \
  --batch-size $BATCH_SIZE \
  --eval-batch-size $EVAL_BATCH_SIZE \
  --num-epochs $NUM_EPOCHS \
  --tuned-layers-count 0 \
  --dropout 0.1 \
| tee results/berkeley-multitarget-nofinetune.txt

python3 multitarget_trainer.py \
  --model-path $BASE_MODEL \
  --cache-dir $CACHE_DIR \
  --data-dir "data/hf_berkeley_multi_wide" \
  --output-dir "test-classifier-berkeley" \
  --model-dir "outputs/berkeley_classifier_multi_wide" \
  --batch-size $BATCH_SIZE \
  --eval-batch-size $EVAL_BATCH_SIZE \
  --num-epochs $NUM_EPOCHS \
  --tuned-layers-count -1 \
  --dropout 0.1 \
| tee results/berkeley-multitarget-finetune.txt

python3 multitarget_trainer.py \
  --model-path "GroNLP/hateBERT" \
  --cache-dir $CACHE_DIR \
  --data-dir "data/hf_berkeley_multi_wide" \
  --output-dir "test-classifier-berkeley" \
  --model-dir "outputs/berkeley_classifier_multi_wide_hatebert" \
  --batch-size $BATCH_SIZE \
  --eval-batch-size $EVAL_BATCH_SIZE \
  --num-epochs $NUM_EPOCHS \
  --tuned-layers-count -1 \
  --dropout 0.1 \
| tee results/berkeley-multitarget-finetune-hatebert.txt

python3 multitarget_trainer.py \
  --model-path $BASE_MODEL \
  --cache-dir $CACHE_DIR \
  --data-dir "data/hf_berkeley_multi_wide" \
  --output-dir "test-classifier-berkeley-lora" \
  --model-dir "outputs/berkeley_classifier_multi_wide_lora" \
  --batch-size $BATCH_SIZE \
  --eval-batch-size $EVAL_BATCH_SIZE \
  --num-epochs $NUM_EPOCHS \
  --tuned-layers-count 0 \
  --dropout 0.1 \
  --use-lora \
  --lora-args '{
        "init_lora_weights": "olora",
        "target_modules": "all-linear"
    }' \
| tee results/berkeley-multitarget-lora.txt


python3 multitask_trainer.py \
  --model-path $BASE_MODEL \
  --cache-dir $CACHE_DIR \
  --data-dir "data/hf_berkeley_multi" \
  --output-dir "test-classifier-berkeley" \
  --model-dir "outputs/berkeley_classifier_multi" \
  --batch-size $BATCH_SIZE \
  --eval-batch-size $EVAL_BATCH_SIZE \
  --num-epochs $NUM_EPOCHS \
  --tuned-layers-count 0 \
  --dropout 0.1 \
| tee results/berkeley-multitask-nofinetune.txt

python3 multitask_trainer.py \
  --model-path $BASE_MODEL \
  --cache-dir $CACHE_DIR \
  --data-dir "data/hf_berkeley_multi" \
  --output-dir "test-classifier-berkeley" \
  --model-dir "outputs/berkeley_classifier_multi" \
  --batch-size $BATCH_SIZE \
  --eval-batch-size $EVAL_BATCH_SIZE \
  --num-epochs $NUM_EPOCHS \
  --tuned-layers-count -1 \
  --dropout 0.1 \
| tee results/berkeley-multitask-finetune.txt

python3 multitarget_trainer.py \
  --model-path "GroNLP/hateBERT" \
  --cache-dir $CACHE_DIR \
  --data-dir "data/hf_berkeley_multi" \
  --output-dir "test-classifier-berkeley" \
  --model-dir "outputs/berkeley_classifier_multi_hatebert" \
  --batch-size $BATCH_SIZE \
  --eval-batch-size $EVAL_BATCH_SIZE \
  --num-epochs $NUM_EPOCHS \
  --tuned-layers-count -1 \
  --dropout 0.1 \
| tee results/berkeley-multitask-finetune-hatebert.txt

python3 multitask_trainer.py \
  --model-path $BASE_MODEL \
  --cache-dir $CACHE_DIR \
  --data-dir "data/hf_berkeley_multi" \
  --output-dir "test-classifier-berkeley-lora" \
  --model-dir "outputs/berkeley_classifier_multi_lora" \
  --batch-size $BATCH_SIZE \
  --eval-batch-size $EVAL_BATCH_SIZE \
  --num-epochs $NUM_EPOCHS \
  --tuned-layers-count 0 \
  --dropout 0.1 \
  --use-lora \
  --lora-args '{
        "init_lora_weights": "olora",
        "target_modules": "all-linear"
    }' \
| tee results/berkeley-multitask-lora.txt
