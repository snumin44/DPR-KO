import sys
import faiss
import torch
import logging
import argparse
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

sys.path.append('../')
from dpr.model import Pooler
from dpr.data_loader import BiEncoderDataset
from database.vector_database import VectorDatabase
from utils.bm25 import BM25Reranker
from utils.utils import get_topk_accuracy

LOGGER = logging.getLogger()

def argument_parser():

    parser = argparse.ArgumentParser(description='get topk-accuracy of retrieval model')
    
    parser.add_argument('--model', type=str, required=True,
                        help='Directory of pretrained encoder model'
                       )
    parser.add_argument('--valid_data', type=str, required=True,
                        help='Path of validation dataset'
                       )
    parser.add_argument('--faiss_path', type=str, required=True,
                        help='Path of faiss pickle'
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
    parser.add_argument('--max_length', default=512, type=int,
                        help='Max length of sequence'
                       )                        
    parser.add_argument('--pooler', default='cls', type=str,
                        help='Pooler type : {pooler_output|cls|mean|max}'
                       )
    parser.add_argument('--padding', action="store_false", default=True,
                        help='Add padding to short sentences'
                       )
    parser.add_argument('--truncation', action="store_false", default=True,
                        help='Truncate extra tokens when exceeding the max_length'
                       )
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size'
                       )
    parser.add_argument('--device', default = 'cuda' if torch.cuda.is_available() else 'cpu', type=str,
                        help = 'Choose a type of device for training'
                       )    
    
    args = parser.parse_args()
    return args

def init_logging():
    LOGGER.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]','%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    LOGGER.addHandler(console)

def search_evaluation(q_encoder, tokenizer, test_dataset, faiss_index, text_index,
                      search_k=2000, bm25_model=None, faiss_weight=1, bm25_weight=0.5, max_length=512, 
                      pooler=None, padding=True, truncation=True, batch_size=32, device='cuda'):
    
    question = test_dataset.question
    answer_idx = test_dataset.answer_idx

    q_encoder = q_encoder.to(device)

    q_encoder.eval()
    
    question_embed = []
    for start_index in tqdm(range(0, len(question), batch_size)):
        batch_question = question[start_index : start_index + batch_size]
           
        q_batch = tokenizer(batch_question,
                            padding=padding,
                            max_length=max_length,
                            truncation=truncation,)
        
        q_encoder.eval()
        with torch.no_grad():
            q_output = q_encoder(input_ids=torch.tensor(q_batch['input_ids']).to(device),
                                 attention_mask=torch.tensor(q_batch['attention_mask']).to(device),
                                 token_type_ids=torch.tensor(q_batch['token_type_ids']).to(device),)
        
        attention_mask = torch.tensor(q_batch['attention_mask'])
        
        if pooler:
            pooler_output = pooler(attention_mask, q_output).cpu()
        else:
            pooler_output = q_output.last_hidden_state[:,0,:].cpu()
        
        question_embed.append(pooler_output)
     
    question_embed = np.vstack(question_embed) 

    print('>>> Searching documents using faiss index.')
    D, I = faiss_index.search(question_embed, search_k)

    if bm25_model:        
        print('>>> Reranking candidates with BM25 scores.')
        bm25_scores = bm25_model.get_bm25_rerank_scores(question, I)
        total_scores = faiss_weight * D + bm25_weight * bm25_scores
            
        for idx in range(total_scores.shape[0]):
            sorted_idx = np.argsort(total_scores[idx])[::-1]
            # D[idx] = D[idx][sorted_idx]
            I[idx] = I[idx][sorted_idx]
        
    scores = get_topk_accuracy(I, answer_idx, text_index)

    return scores


def main(args):
    init_logging()
    
    LOGGER.info('*** Top-k Retrieval Accuracy ***')
    
    # Load model & tokenizer
    q_encoder = AutoModel.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    pooler = Pooler(args.pooler)
    
    # Load valid dataset.
    test_dataset = BiEncoderDataset.load_valid_dataset(args.valid_data)

    # Load faiss index.
    faiss_vector = VectorDatabase(args.faiss_path)
    
    faiss_index = faiss_vector.faiss_index
    text_index = faiss_vector.text_index
    
    # Load bm25 model.
    if args.bm25_path:
        bm25_model = BM25Reranker(bm25_pickle=args.bm25_path)
    else:
        bm25_model = None

    # Get top-k accuracy
    scores = search_evaluation(q_encoder, tokenizer, test_dataset, faiss_index, text_index, search_k=args.search_k,
                               bm25_model=bm25_model, faiss_weight=args.faiss_weight, bm25_weight=args.bm25_weight,
                               max_length=args.max_length, pooler=pooler, padding=args.padding, truncation=args.truncation,
                               batch_size=args.batch_size, device=args.device)

    print()
    print('=== Top-k Accuracy ===')
    print(f"Top1 Acc: {scores['top1_accuracy']*100:.2f} (%)")
    print(f"Top5 Acc: {scores['top5_accuracy']*100:.2f} (%)")
    print(f"Top10 Acc: {scores['top10_accuracy']*100:.2f} (%)")
    print(f"Top20 Acc: {scores['top20_accuracy']*100:.2f} (%)")
    print(f"Top50 Acc: {scores['top50_accuracy']*100:.2f} (%)")
    print(f"Top100 Acc: {scores['top100_accuracy']*100:.2f} (%)")
    print('======================')

if __name__ == '__main__':
    args = argument_parser()
    main(args)
