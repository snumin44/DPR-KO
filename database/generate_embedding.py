import os
import sys
import torch
import random
import logging
import argparse
import numpy as np
from transformers import AutoModel, AutoTokenizer

from vector_database import VectorDatabase

sys.path.append('../')
from dpr.data_loader import BiEncoderDataset
from dpr.model import Pooler
from utils.bm25 import BM25Reranker

LOGGER = logging.getLogger()

def argument_parser():

    parser = argparse.ArgumentParser(description='Build vector database with wiki text')

    parser.add_argument('--model', type=str, required=True,
                        help='Directory of pretrained encoder model'
                       )
    parser.add_argument('--wiki_path', type=str, default='../wikidump/text',
                        help='Path of wiki dump'
                       )
    parser.add_argument('--valid_data', type=str, required=False,
                        help='Path of validation dataset'
                       )
    parser.add_argument('--save_path', type=str, default='../pickles',
                        help='Save directory of faiss index'
                       )
    parser.add_argument('--save_context', action='store_true', default=False,
                        help='Save text and title with faiss index'
                       )
    parser.add_argument('--train_bm25', action='store_true', default=False,
                        help='Train bm25 with the same corpus'
                       )
    parser.add_argument('--num_sent', type=int, default=5,
                        help='Number of sentences consisting of a wiki chunk'
                       )
    parser.add_argument('--overlap', type=int, default=0,
                        help='Number of overlapping sentences between consecutive chunks'
                       )
    parser.add_argument('--pooler', default='cls', type=str,
                        help='Pooler type : {pooler_output|cls|mean|max}'
                       )       
    parser.add_argument('--max_length', type=int, default=512,
                        help='Max length for encoder model'
                       )
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size'
                       )
    parser.add_argument('--cpu_workers', type=int, required=False,
                        help='Number of cpu cores used in chunking wiki text'
                       ) 
    parser.add_argument('--device', default = 'cuda' if torch.cuda.is_available() else 'cpu', type=str,
                        help = 'Choose a type of device for training'
                       )
    parser.add_argument('--random_seed', default = 42, type=int,
                        help = 'Random seed'
                       ) 

    args = parser.parse_args()
    return args

    
def init_logging():
    LOGGER.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]','%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    LOGGER.addHandler(console)


def seed_everything(args):
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    os.environ["PYTHONHASHSEED"] = str(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):
    init_logging()
    seed_everything(args)

    LOGGER.info('*** Building Vector Database ***')
    
    vector_db = VectorDatabase()
    
    if args.valid_data:
        valid_dataset = BiEncoderDataset.load_valid_dataset(args.valid_data)
        gold_passages = valid_dataset.positive_ctx
        print(args.cpu_workers)
    else:
        gold_passages = None
        
    model = AutoModel.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    pooler = Pooler(args.pooler)
    
    vector_db.build_embedding(wiki_path=args.wiki_path,
                              save_path=args.save_path,
                              save_context=args.save_text,
                              tokenizer=tokenizer,
                              embedding_model=model,
                              pooler=pooler,
                              cpu_workers=args.cpu_workers,
                              gold_passages=gold_passages,
                              device = args.device,
                             )

    #### Train BM 25 ####
    if args.train_bm25:
        bm25_model = BM25Reranker()
        bm25_model.build_bm25_model(text=vector_db.text,
                                    title=vector_db.title,
                                    path=args.save_path)
        
if __name__ == '__main__':
    args = argument_parser()
    main(args)  