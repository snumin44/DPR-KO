#!/bin/sh

# If you wanna use more gpus, set GPU_ID like "0, 1, 2"

GPU_ID=0
CUDA_VISIBLE_DEVICES=$GPU_ID python3 train.py \
    	--model 'klue/bert-base' \
    	--train_data '../data/korquad_v1_train.json' \
        --valid_data '../data/korquad_v1_valid.json' \
        --q_output_path '../pretrained_model/question_encoder' \
        --c_output_path '../pretrained_model/context_encoder' \
    	--epochs 10 \
        --batch_size 64 \
        --max_length 512 \
        --dropout 0.1 \
        --pooler 'cls' \
        --amp \