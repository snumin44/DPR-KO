import os
import json
import asyncio
import pickle
import torch
import faiss
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from nltk import sent_tokenize


class VectorDatabase(object):
    
    def __init__(self, faiss_pickle=None, context_pickle=None):
        self.text = None
        self.title = None
        self.text_index = None
        self.faiss_index = None
        
        if faiss_pickle:
            self._load_faiss_pickle(faiss_pickle)

        if context_pickle:
            self._load_context_pickle(context_pickle)
        
    def _load_faiss_pickle(self, faiss_pickle):
        print('>>> Loading faiss index.')
        with open(faiss_pickle, 'rb') as file:
            data = pickle.load(file)
            self.text_index = data.get('text_index', None)
            self.faiss_index = data.get('faiss_index', None)

    def _load_context_pickle(self, context_pickle):
        print('>>> Loading text and title.')
        with open(context_pickle, 'rb') as file:
            data = pickle.load(file)
            self.text = data.get('text', None)
            self.title = data.get('title', None) 
    
    def _process_file(self, file_path):
        with open(file_path, "rt", encoding="utf8") as f:
             input_txt = f.read().strip()
        return input_txt.split("</doc>")   
    
    def _process_text(self, arguments):
        txt, num_sent, overlap, test_titles = arguments
        txt = txt.strip()
        if not txt:
            return [], [], 0

        lines = txt.split("\n")
        title = lines[0].strip(">").split("title=")[1].strip('"')
        text = " ".join(lines[2:]).strip().replace('()', '').replace('\n', ' ').replace('(, ', '(')
        
        if len(text.split()) <= 10:
            return [], [], 0

        if title in test_titles:
            return [], [], 1
        
        txt_lst, title_lst = [], []
        start, end = 0, num_sent
        total_sents = sent_tokenize(text)

        while start < len(total_sents):
            chunk = total_sents[start:end]
            txt_lst.append(' '.join(chunk))
            title_lst.append(title)
            start += (num_sent - overlap)
            end = start + num_sent

        return txt_lst, title_lst, 0

    def _load_wikidata_by_chunk(self, wiki_path, num_sent=5, overlap=0, cpu_workers=None, gold_passages=None):
        print('>>> Loading wiki data.')
    
        wiki_subsets = os.listdir(wiki_path)
        wiki_input_txt = []
    
        if not cpu_workers or cpu_workers < 1:
            cpu_workers = os.cpu_count()
                
        # Use multiprocessing to read files in parallel
        with Pool(cpu_workers) as pool:
            file_paths = [os.path.join(wiki_path, subset, wiki_file)
                          for subset in wiki_subsets
                          for wiki_file in os.listdir(os.path.join(wiki_path, subset))]
            for result in tqdm(pool.imap_unordered(self._process_file, file_paths), total=len(file_paths)):
                wiki_input_txt.extend(result)
    
        idx_lst, txt_lst, title_lst = [], [], [] 
    
        # Store text from test set first.
        if gold_passages:
            for ctx in gold_passages:
                if ctx['idx'] not in idx_lst:
                    idx_lst.append(ctx['idx'])
                    txt_lst.append(ctx['text'])
                    title_lst.append(ctx['title'].replace('_', ' '))

        chunk_idx = -1 if len(idx_lst) == 0 else max(idx_lst)

        print('>>> Parsing and chunking wiki data.')

        # Use multiprocessing to process text chunks in parallel
        tasks = [(txt, num_sent, overlap, title_lst) for txt in wiki_input_txt]
        total_duplicates = 0
        
        txt_temp, title_temp = [], []        
        with Pool(cpu_workers) as pool:    # chunk_idx_results
            for txt_results, title_results, duplicates in tqdm(pool.imap_unordered(self._process_text, tasks), total=len(tasks)):
                txt_temp.extend(txt_results)
                title_temp.extend(title_results)
                total_duplicates += duplicates 

        txt_lst.extend(txt_temp)
        title_lst.extend(title_temp)
        
        if chunk_idx < len(txt_lst):
            for jdx in range(chunk_idx + 1, len(txt_lst)):
                idx_lst.append(jdx)

        # print(len(idx_lst))
        # print(len(txt_lst))
        # print(len(title_lst))
        
        if gold_passages:
            print(f'>>> Deleted {total_duplicates} documents that were duplicates of gold passages.')
        
        return idx_lst, txt_lst, title_lst

    def encode_text(self, title, text, embedding_model, tokenizer, pooler=None, max_length=512, batch_size=32, device='cuda'):
    
        print('>>> Encoding wiki data.')

        embedding_model = embedding_model.to(device)
        
        all_ctx_embed = []

        embedding_model.eval()
        for start_index in tqdm(range(0, len(text), batch_size)):
            batch_txt = text[start_index : start_index + batch_size]
            batch_title = title[start_index : start_index + batch_size]
                           
            batch = tokenizer(batch_title,
                              batch_txt,
                              padding=True,
                              truncation=True,
                              max_length=max_length,)

            with torch.no_grad():
                output = embedding_model(input_ids=torch.tensor(batch['input_ids']).to(device),
                                        attention_mask=torch.tensor(batch['attention_mask']).to(device),
                                        token_type_ids=torch.tensor(batch['token_type_ids']).to(device),)

                attention_mask = torch.tensor(batch['attention_mask']).to(device)
                
                if pooler:
                    pooler_output = pooler(attention_mask, output) 
                else:
                    pooler_output = output.last_hidden_state[:,0,:]
        
            all_ctx_embed.append(pooler_output.cpu())
     
        all_ctx_embed = np.vstack(all_ctx_embed) 
        
        return all_ctx_embed    
    
    def build_embedding(self,
                        wiki_path=None,
                        save_path=None,
                        save_context=None,
                        tokenizer=None,
                        embedding_model=None,
                        pooler = None,
                        num_sent=5,
                        overlap=0,
                        cpu_workers=None,
                        gold_passages=None,
                        embedding_size=768,
                        max_length=512,
                        batch_size=32,
                        device='cuda',
                        ):
        
        idx_lst, txt_lst, title_lst = self._load_wikidata_by_chunk(wiki_path, num_sent, overlap, cpu_workers, gold_passages)
                
        all_embeddings = self.encode_text(title_lst, txt_lst, embedding_model, tokenizer, pooler, max_length, batch_size, device)
        
        faiss.normalize_L2(all_embeddings)
        faiss_index = faiss.IndexFlatIP(embedding_size)
        faiss_index.add(all_embeddings)
        
        print(">>> Saving faiss pickle. It contains \'text_index\' and \'faiss_index\'.")  
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        pickle_file_path = os.path.join(save_path, 'faiss_pickle.pkl')
    
        with open(pickle_file_path, 'wb') as file:
            pickle.dump({
                'text_index': idx_lst,
                'faiss_index':faiss_index,
            }, file)

        if save_context:
            print(">>> Saving context pickle. It contains \'title\' and \'text\'.")
            pickle_file_path = os.path.join(save_path, 'context_pickle.pkl')
            with open(pickle_file_path, 'wb') as file:
                pickle.dump({
                    'title': title_lst,
                    'text':txt_lst,
                }, file)
                    
        self.text = txt_lst
        self.title = title_lst
        self.text_index = idx_lst
        self.faiss_index = faiss_index

        print(f'>>> Total number of passages: {len(self.text_index)}')