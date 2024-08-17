import faiss
import sys
import torch
import argparse
import numpy as np
from transformers import (
        AutoModel,
        AutoTokenizer,
        AutoConfig,
        AutoModelForCausalLM,
        TextStreamer,
)

sys.path.append('../')
from dpr.model import Pooler
from database.vector_database import VectorDatabase
from utils.bm25 import BM25Reranker
from utils.utils import get_topk_accuracy

def argument_parser():

    parser = argparse.ArgumentParser(description='generation based on retrieval results')

    # Encoder Model
    parser.add_argument('--search_model', type=str, required=True,
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

    # Generative Model
    parser.add_argument('--generative_model', type=str, required=True,
                        help='Directory of pretrained generative model'
                       )    
    parser.add_argument('--max_new_tokens', default=128, type=int,
                        help='The maximum numbers of tokens to generate'
                       )
    parser.add_argument('--num_beams', default=5, type=int,
                        help='Number of beams for beam search. 1 means no beam search.'
                       )
    parser.add_argument('--do_sample', action='store_true', default=False,
                        help='Whether or not to use sampling'
                       )
    parser.add_argument('--temperature', default=1.0, type=float,
                        help='Number of beams for beam search. 1 means no beam search.'
                       )
    parser.add_argument('--top_k', default=50, type=int,
                        help='Number of highest probability vocabulary tokens to keep for top-k-filtering'
                       )
    parser.add_argument('--top_p', default=1.0, type=float,
                        help=' If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.'
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

class Generator(object):
    
    PROMPT = """
            주어진 텍스트를 참고해 질문에 대답해주세요. 질문과 관련이 없는 내용은 참고하지 않습니다.\
            ### 질문: {0}\
            ### 텍스트: {1}\
            ### 답변:
            """
    def __init__(self, model):
        
        self.model = AutoModelForCausalLM.from_pretrained(model)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

    def generate(self, prompt, max_new_tokens=128, num_beams=5, do_sample=False, temperature=1.0, top_k=50, top_p=1):
        self.model.eval()
        result = self.model.generate(
            **self.tokenizer(
                prompt,
                return_tensors='pt',
                return_token_type_ids=False),
            streamer = self.streamer,
            pad_token_id= self.tokenizer.eos_token_id,
            
            max_new_tokens=max_new_tokens,
            num_bemas = num_beams,
            do_sample= do_sample,
            temperature=temperature,
            top_k = top_k,
            top_p = top_p,
        )


def main(args):
    
    # Load encoder model & tokenizer
    q_encoder = AutoModel.from_pretrained(args.search_model)
    tokenizer = AutoTokenizer.from_pretrained(args.search_model)
    config = AutoConfig.from_pretrained(args.search_model)
    
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

    # Load generative model & tokenizer
    g_decoder = Generator(args.generative_model)
    
    # Retrieval loop.
    while True:
        input_text = input('What would you like to know from Korean Wikipedia? (type "exit" to quit): ')
        
        if input_text.lower() == "exit":
            print("Exiting the inference loop.")
            break
        
        D, I = inference(input_text, q_encoder, tokenizer, faiss_index, text, title,
                         search_k=args.search_k, return_k=args.return_k, bm25_model=bm25_model, faiss_weight=args.faiss_weight,
                         bm25_weight=args.bm25_weight, max_length=args.max_length, pooler=pooler, truncation=args.truncation, device=args.device)
            

        retrieved_text, retrieved_title = [], []
        for idx, (distance, index) in enumerate(zip(D[0], I[0])):
            retrieved_text.append(f'[{idx+1}]' + text[index])
            if title[index] not in retrieved_title:
                retrieved_title.append(title[index])

            if idx + 1 == args.return_k:
                break
        
        prompt = g_decoder.PROMPT.format(input_text, ' '.join(retrieved_text))
        
        g_decoder.generate(prompt, args.max_new_tokens, args.num_beams, args.do_sample, args.temperature, args.top_k, args.top_p)
        print()
        print(f"※ 참고문헌: 위키피디아 \'{retrieved_title[0]}\' 등")
        print()
            

if __name__ == '__main__':
    args = argument_parser()
    main(args)
