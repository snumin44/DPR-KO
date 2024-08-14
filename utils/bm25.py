import os
import sys
import pickle
import logging
import argparse
import numpy as np
from rank_bm25 import BM25Okapi

sys.path.append('../')
from dpr.data_loader import BiEncoderDataset
from database.vector_database import VectorDatabase

LOGGER = logging.getLogger()

def argument_parser():

    parser = argparse.ArgumentParser(description='train bm25 model')

    parser.add_argument('--bm25_corpus', type=str, default='wiki',
                        help='Type of training dataset: {wiki|all}'
                       )
    parser.add_argument('--valid_data', type=str, default='../data/korquad_v1_valid.json',
                        help='Path of validation dataset '
                       )
    parser.add_argument('--wiki_path', type=str, default='../wikidump/text',
                        help='Path of validation dataset '
                       )
    parser.add_argument('--save_path', type=str, default='../pickles',
                        help='Save directory of question encoder'
                       )
    parser.add_argument('--num_sent', type=int, default=5,
                        help='Number of sentences consisting of a wiki chunk'
                       )
    parser.add_argument('--overlap', type=int, default=0,
                        help='Number of overlapping sentences between consecutive chunks'
                       )
    parser.add_argument('--cpu_workers', type=int, required=False,
                        help='Number of cpu cores used in chunking wiki text'
                       ) 

    args = parser.parse_args()
    return args

def init_logging():
    LOGGER.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]','%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    LOGGER.addHandler(console)


class BM25Reranker(object):

    def __init__(self, tokenizer=None, bm25_pickle=None):
        self.model = None
        self.tokenizer = tokenizer=None
        
        if bm25_pickle:
            self._load_bm25_pickle(bm25_pickle)

    def _load_bm25_pickle(self, bm25_pickle):
        print('>>> Loading BM25 model.')
        with open(bm25_pickle, 'rb') as file:
            self.model = pickle.load(file)

    def _save_bm25_pickle(self, model, path):
        if not os.path.exists(path):
            os.makedirs(path)

        pickle_file_path = os.path.join(path, 'bm25_pickle.pkl')
        with open(pickle_file_path, 'wb') as file:
            pickle.dump(model, file)
            
    def _tokenize(self, text):
        tokenized_text = [txt.split() for txt in text] 
        return tokenized_text

    def _prepend_title_to_text(self, text, title):
        text_with_title = []
        for _text, _title in zip(text, title):
            text_with_title.append(_title + '. ' + _text)
        return text_with_title
        
    def build_bm25_model(self, text, title=None, path='./bm25_model'): 
        if title:
            prepended_text = self._prepend_title_to_text(text, title)
            tokenized_text = self._tokenize(prepended_text) 
        else:
            tokenized_text = self._tokenize(text)
            
        print('>>> Training BM25 model...')        
        model = BM25Okapi(tokenized_text) 

        print('>>> Training done.')
        self.model = model
        self._save_bm25_pickle(model, path)

    def get_bm25_rerank_scores(self, questions, doc_ids):
        tokenized_questions = self._tokenize(questions)    
        bm25_scores = []
        for question, doc_id in zip(tokenized_questions, doc_ids):
            # bm25_score : [0.        , 0.93729472, 0.        ... ]
            bm25_score = self.model.get_batch_scores(question, doc_id)
            bm25_scores.append(bm25_score)

        return np.array(bm25_scores)

def main(args):
    
    init_logging()

    LOGGER.info('*** BM25 Training ***')

    if not args.cpu_workers or args.cpu_workers < 1:
        cpu_workers = os.cpu_count()
    else:
        cpu_workers = args.cpu_workers
    
    if args.bm25_corpus == 'wiki':
        valid_dataset = BiEncoderDataset.load_valid_dataset(args.valid_data)
        gold_passages = valid_dataset.positive_ctx
    
        vector_db = VectorDatabase()
        _, text, title = vector_db._load_wikidata_by_chunk(args.wiki_path,
                                                           num_sent=args.num_sent,
                                                           overlap=args.overlap,
                                                           cpu_workers=cpu_workers,
                                                           gold_passages=gold_passages)
    
    elif args.bm25_corpus == 'all':
        vector_db = VectorDatabase()
        _, text, title = vector_db._load_wikidata_by_chunk(args.wiki_path,
                                                           num_sent=args.num_sent,
                                                           overlap=args.overlap,
                                                           cpu_workers=args.cpu_workers,
                                                           gold_passages=None)
        
    else:
        raise ValueError(f"Invalid bm25_corpus value: {args.bm25_corpus}. Expected 'wiki' or 'all'.")

    bm25_model = BM25Reranker()
    bm25_model.build_bm25_model(text, title, args.save_path)


if __name__ == '__main__':
    args = argument_parser()
    main(args)
