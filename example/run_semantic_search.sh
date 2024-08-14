#!/bin/sh

# If you wanna use more gpus, set GPU_ID like "0, 1, 2"

GPU_ID=0
CUDA_VISIBLE_DEVICES=$GPU_ID python3 semantic_search.py \
    	--model '../pretrained_model/question_encoder' \
        --faiss_path '../pickles/faiss_pickle.pkl' \
        --context_path '../pickles/context_pickle.pkl' \
        --bm25_path '../pickles/bm25_pickle.pkl' \
        --faiss_weight 1 \
        --bm25_weight 0.3 \
        --search_k 2000 \
        --return_k 5 \
        --pooler 'cls' \
