#!/bin/sh

# If you wanna use more gpus, set GPU_ID like "0, 1, 2"

GPU_ID=0
CUDA_VISIBLE_DEVICES=$GPU_ID python3 generate_embedding.py \
    	--model '../pretrained_model/context_encoder' \
        --wiki_path '../wikidump/text' \
        --valid_data '../data/korquad_v1_valid.json' \
        --save_path '../pickles' \
        --save_text \
        --train_bm25 \
        --pooler 'cls' \
        --num_sent 5 \
        --overlap 0 \
        --max_length 512 \
        --batch_size 128 \
        --cpu_workers 50
