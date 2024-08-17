#!/bin/sh

# If you wanna use more gpus, set GPU_ID like "0, 1, 2"

GPU_ID="0, 1, 2"
CUDA_VISIBLE_DEVICES=$GPU_ID python3 generation.py \
    	--search_model '../pretrained_model/question_encoder' \
        --generative_model 'beomi/KoAlpaca-Polyglot-5.8B'\
        --faiss_path '../pickles/faiss_pickle.pkl' \
        --bm25_path '../pickles/bm25_pickle.pkl'\
        --context_path '../pickles/context_pickle.pkl' \
        --faiss_weight 1 \
        --bm25_weight 0.3 \
        --search_k 2000 \
        --return_k 3 \
        --pooler 'cls'
        
