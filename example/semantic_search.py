import faiss
import sys
import torch
import argparse
import numpy as np
from transformers import AutoModel, AutoTokenizer, AutoConfig 

sys.path.append('../')
from dpr.model import Pooler
from database.vector_database import VectorDatabase
from utils.bm25 import BM25Reranker
from utils.utils import get_topk_accuracy

def argument_parser():

    parser = argparse.ArgumentParser(description='get topk-accuracy of retrieval model')
    
    parser.add_argument('--model', type=str, required=True,
                        help='Directory of pretrained encoder model'
                       )
    parser.add_argument('--faiss_path', type=str, required=True,
                        help='Path of faiss pickle'
                       )
    parser.add_argument('--context_path', type=str, required=True,
                        help='Path of context pickle'
                       )
    parser.add_argument('--bm25_path', type=str, required=False,
                        help='Path of BM25 Model'
                       )
    parser.add_argument('--faiss_weight', default=1, type=float, 
                        help='Weight for semantic search'
                       )
    parser.add_argument('--bm25_weight', default=0.5, type=float, 
                        help='Weight for BM25 rerank score'
                       )
    parser.add_argument('--search_k', default=2000, type=int,
                        help='Number of retrieved documents'
                       )
    parser.add_argument('--return_k', default=5, type=int,
                        help='Number of returned documents'
                       )   
    parser.add_argument('--max_length', default=512, type=int,
                        help='Max length of sequence'
                       )                        
    parser.add_argument('--pooler', default='cls', type=str,
                        help='Pooler type : {pooler_output|cls|mean|max}'
                       )
    parser.add_argument('--truncation', action="store_false", default=True,
                        help='Truncate extra tokens when exceeding the max_length'
                       )
    parser.add_argument('--device', default = 'cuda' if torch.cuda.is_available() else 'cpu', type=str,
                        help = 'Choose a type of device for training'
                       )    
    
    args = parser.parse_args()
    return args


def inference(question, q_encoder, tokenizer, faiss_index, text, title,
              search_k=2000, return_k=5, bm25_model=None, faiss_weight=1, bm25_weight=0.5,
              max_length=512, pooler=None, truncation=True, device='cuda'):
    
    q_encoder = q_encoder.to(device)
               
    features = tokenizer(question, max_length=max_length, truncation=truncation, return_tensors='pt').to(device)

    q_encoder.eval()
    with torch.no_grad():
        q_output = q_encoder(**features, return_dict=True)

    pooler_output = pooler(features['attention_mask'], q_output)
    pooler_output = pooler_output.cpu().detach().numpy() # (1, 768)

    D, I = faiss_index.search(pooler_output, search_k)

    if bm25_model:        
        bm25_scores = bm25_model.get_bm25_rerank_scores(question, I)
        total_scores = faiss_weight * D + bm25_weight * bm25_scores
            
        sorted_idx = np.argsort(total_scores[0])[::-1]
        D[0] = D[0][sorted_idx]
        I[0] = I[0][sorted_idx]

    return D, I


def main(args):
    
    # Load model & tokenizer
    q_encoder = AutoModel.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    config = AutoConfig.from_pretrained(args.model)
    
    pooler = Pooler(args.pooler)
    
    # Load faiss index.
    faiss_vector = VectorDatabase(args.faiss_path, args.context_path)

    text = faiss_vector.text
    title = faiss_vector.title
    faiss_index = faiss_vector.faiss_index

    # Load bm25 model.
    if args.bm25_path:
        bm25_model = BM25Reranker(bm25_pickle=args.bm25_path)
    else:
        bm25_model = None

    # Retrieval loop.
    while True:
        input_text = input('What would you like to know from Korean Wikipedia? (type "exit" to quit): ')
        
        if input_text.lower() == "exit":
            print("Exiting the inference loop.")
            break
        
        D, I = inference(input_text, q_encoder, tokenizer, faiss_index, text, title,
                         search_k=args.search_k, return_k=args.return_k, bm25_model=bm25_model, faiss_weight=args.faiss_weight,
                         bm25_weight=args.bm25_weight, max_length=args.max_length, pooler=pooler, truncation=args.truncation, device=args.device)
            
        for idx, (distance, index) in enumerate(zip(D[0], I[0])):
            print()
            print(f'|| Retrieval Ranking: {idx+1} || Similarity Score: {distance:.2f} || Title: {title[index]} ||')
            print('================================================================================')
            print(text[index].replace('\n', ' '))
            print()

            if idx + 1 == args.return_k:
                break


if __name__ == '__main__':
    args = argument_parser()
    main(args)
