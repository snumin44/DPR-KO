import os
import re
import sys
import json
import time
import faiss
import random
import logging
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
from transformers import AutoModel, AutoTokenizer, AutoConfig
from transformers import get_linear_schedule_with_warmup

sys.path.append("../") 
from dpr.model import Pooler, BiEncoder
from dpr.data_loader import BiEncoderDataset, DataCollator 

from utils.utils import format_time, get_topk_accuracy   

LOGGER = logging.getLogger()

def argument_parser():

    parser = argparse.ArgumentParser(description='train dense passage retrieval model')

    parser.add_argument('--model', type=str, required=True,
                        help='Directory of pretrained model'
                       )    
    parser.add_argument('--train_data', type=str, required=True,
                        help='Path of training dataset '
                       )
    parser.add_argument('--valid_data', type=str, required=True,
                        help='Path of validation dataset '
                       )
    parser.add_argument('--q_output_path', type=str, default='../pretrained_model/question_encoder',
                        help='Save directory of question encoder'
                       )
    parser.add_argument('--c_output_path', type=str, default='../pretrained_model/question_encoder',
                        help='Save directory of context encoder'
                       )

    # Tokenizer & Collator settings
    parser.add_argument('--max_length', default=512, type=int,
                        help='Max length of sequence'
                       )
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size'
                       )
    parser.add_argument('--padding', action="store_false", default=True,
                        help='Add padding to short sentences'
                       )
    parser.add_argument('--truncation', action="store_false", default=True,
                        help='Truncate extra tokens when exceeding the max_length'
                       )
    parser.add_argument('--shuffle', action="store_false", default=True,
                        help='Load shuffled sequences'
                       )

    # Train config    
    parser.add_argument('--epochs', default=15, type=int,
                        help='Training epochs'
                       )      
    parser.add_argument('--eval_epoch', default=1, type=int,
                        help='Epoch for evaluation'
                       )      
    parser.add_argument('--early_stop_epoch', default=5, type=int,
                        help='Epoch for eearly stopping'
                       )          
    parser.add_argument('--pooler', default='cls', type=str,
                        help='Pooler type : {pooler_output|cls|mean|max}'
                       )    
    parser.add_argument('--weight_decay', default=1e-2, type=float,
                        help='Weight decay'
                       )       
    parser.add_argument('--no_decay', nargs='+', default=['bias', 'LayerNorm.weight'],
                        help='List of parameters to exclude from weight decay' 
                       )         
    parser.add_argument('--temp', default=0.05, type=float,
                        help='Temperature for similarity'
                       )       
    parser.add_argument('--dropout', default=0.1, type=float,
                        help='Drop-out ratio'
                       )       
    parser.add_argument('--learning_rate', default=5e-5, type=float,
                        help='Leraning rate'
                       )       
    parser.add_argument('--eta_min', default=0, type=int,
                        help='Eta min for CosineAnnealingLR scheduler'
                       )   
    parser.add_argument('--eps', default=1e-8, type=float,
                        help='Epsilon for AdamW optimizer'
                       )   
    parser.add_argument('--amp', action="store_true",
                        help='Use Automatic Mixed Precision for training'
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


def get_adamw_optimizer(q_encoder, c_encoder, args):
    if args.no_decay: 
        # skip weight decay for some specific parameters i.e. 'bias', 'LayerNorm.weight'.
        no_decay = args.no_decay  
        optimizer_grouped_parameters = [
            {'params': [p for n, p in q_encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in q_encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
            {'params': [p for n, p in c_encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in c_encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},    
        ]
    else:
        # weight decay for every parameter.
        optimizer_grouped_parameters = [q_encoder.parameters()] + [c_encoder.parameters()]
        
    optimizer = AdamW(optimizer_grouped_parameters, lr = args.learning_rate, eps = args.eps)
    return optimizer


def get_scheduler(optimizer, args):
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.eta_min, last_epoch=-1)
    return scheduler


def train(biencoder, train_dataloader, optimizer, scheduler, scaler, args):
 
    total_train_loss = 0
    
    biencoder.train()
    for step, (q_batch, c_batch) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        
        q_input_ids = q_batch['input_ids'].to(args.device)
        q_attn_mask = q_batch['attention_mask'].to(args.device)
        q_token_ids = q_batch['token_type_ids'].to(args.device)        
        
        c_input_ids = c_batch['input_ids'].to(args.device)
        c_attn_mask = c_batch['attention_mask'].to(args.device)
        c_token_ids = c_batch['token_type_ids'].to(args.device)
        
        # pass the data to device(cpu or gpu)            
        optimizer.zero_grad()

        if args.amp:
            train_loss = biencoder(q_input_ids, q_attn_mask, q_token_ids,
                                   c_input_ids, c_attn_mask, c_token_ids,)

            scaler.scale(train_loss.mean()).backward()
            # Clip the norm of the gradients to 5.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(biencoder.parameters(), max_norm=5.0)          
            
            scaler.step(optimizer)
            scaler.update()

        else:
            train_loss = biencoder(q_input_ids, q_attn_mask, q_token_ids,
                                   c_input_ids, c_attn_mask, c_token_ids,)
            
            train_loss.mean().backward()
            # Clip the norm of the gradients to 5.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(biencoder.parameters(), max_norm=5.0)  

            optimizer.step()
        
        scheduler.step()        
        
        total_train_loss += train_loss.mean()
                    
    train_loss = total_train_loss / len(train_dataloader)
    
    return train_loss


def evaluate(biencoder, tokenizer, test_dataset, embedding_size, args):
    
    question = test_dataset.question
    answer_idx = test_dataset.answer_idx

    positive_ctx = test_dataset.positive_ctx
    hard_neg_ctx = test_dataset.hard_neg_ctx
            
    positive_idx, positive_txt, positive_title = [], [], []        

    for idx in range(len(positive_ctx)): 
        positive_idx.append(positive_ctx[idx]['idx'])
        positive_txt.append(positive_ctx[idx]['text'])
        positive_title.append(positive_ctx[idx]['title'])

    all_ctx_embed = []
    
    biencoder.eval()
    for start_index in tqdm(range(0, len(positive_txt), args.batch_size)):
        batch_txt = positive_txt[start_index : start_index + args.batch_size]
        batch_title = positive_title[start_index : start_index + args.batch_size]
           
        c_batch = tokenizer(batch_title,
                            batch_txt,
                            padding=args.padding,
                            max_length=args.max_length,
                            truncation=args.truncation,)
        
        with torch.no_grad():
            pooler_output = biencoder.get_c_embeddings(input_ids=torch.tensor(c_batch['input_ids']).to(args.device),
                                                       attention_mask=torch.tensor(c_batch['attention_mask']).to(args.device),
                                                       token_type_ids=torch.tensor(c_batch['token_type_ids']).to(args.device),)
        all_ctx_embed.append(pooler_output.cpu())
     
    all_ctx_embed = np.vstack(all_ctx_embed) 
    
    faiss.normalize_L2(all_ctx_embed)       
    index = faiss.IndexFlatIP(embedding_size)
    index.add(all_ctx_embed)
    #faiss.write_index(index, 'evaluation.index')

    question_embed = []
    for start_index in tqdm(range(0, len(question), args.batch_size)):
        batch_question = question[start_index : start_index + args.batch_size]
           
        q_batch = tokenizer(batch_question,
                            padding=args.padding,
                            max_length=args.max_length,
                            truncation=args.truncation,)
        
        biencoder.eval()
        with torch.no_grad():
            pooler_output = biencoder.get_q_embeddings(input_ids=torch.tensor(q_batch['input_ids']).to(args.device),
                                                       attention_mask=torch.tensor(q_batch['attention_mask']).to(args.device),
                                                       token_type_ids=torch.tensor(q_batch['token_type_ids']).to(args.device),)
        question_embed.append(pooler_output.cpu())
     
    question_embed = np.vstack(question_embed) 

    D, I = index.search(question_embed, k=100)

    scores = get_topk_accuracy(I, answer_idx, positive_idx)

    return scores


def main(args):

    init_logging()
    seed_everything(args)

    LOGGER.info('*** Bi-Encoder Training ***')
    
    train_dataset = BiEncoderDataset.load_train_dataset(args.train_data)
    valid_dataset = BiEncoderDataset.load_valid_dataset(args.valid_data)

    collator = DataCollator(args)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=args.shuffle, collate_fn=collator)

    model = AutoModel.from_pretrained(args.model)
    config = AutoConfig.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, config=config)

    if args.device == 'cuda':
        biencoder = BiEncoder(args).to(args.device)
        biencoder = torch.nn.DataParallel(biencoder)
        optimizer = get_adamw_optimizer(biencoder.module.q_encoder,
                                        biencoder.module.c_encoder, args)
        LOGGER.info("Using nn.DataParallel")
    
    else:
        biencoder = BiEncoder(args)
        optimizer = get_adamw_optimizer(biencoder.q_encoder,
                                        biencoder.c_encoder, args)
    
    scheduler = get_scheduler(optimizer, args)

    if args.amp:
        scaler = GradScaler()
    else:
        scaler = None
    
    best_score = None
    early_stop_score = list()

    t0 = time.time()
    
    for epoch_i in range(args.epochs):
        
        LOGGER.info(f'Epoch : {epoch_i+1}/{args.epochs}')

        train_loss = train(biencoder, train_dataloader, optimizer, scheduler, scaler, args)
        
        print(f'Epoch:{epoch_i+1}, Train Loss:{train_loss.mean():.4f}')
        
        if epoch_i % args.eval_epoch == 0:

            LOGGER.info('*** Bi-Encoder Evaluation ***')
        
            if isinstance(biencoder, torch.nn.DataParallel):
                model_to_save = biencoder.module   
            else: model_to_save = biencoder
            
            valid_scores = evaluate(model_to_save, tokenizer, valid_dataset, config.hidden_size, args)
        
            top1_acc = valid_scores['top1_accuracy']
            top5_acc = valid_scores['top5_accuracy']
            
            print(f'Epoch:{epoch_i+1}, Top1_Acc:{top1_acc*100:.2f}%, Top5_Acc:{top5_acc*100:.2f}%')
        
            # Check Best Model
            if not best_score or top1_acc > best_score:
                best_score = top1_acc
            
                if not os.path.exists(args.q_output_path):
                    os.makedirs(args.q_output_path)
            
                if not os.path.exists(args.c_output_path):
                    os.makedirs(args.c_output_path)            
            
                model_to_save.save_model(args.q_output_path, args.c_output_path)
    
                LOGGER.info(f'>>> Saved Best Model (Question Encoder) at {args.q_output_path}')
                LOGGER.info(f'>>> Saved Best Model (Context Encoder) at {args.c_output_path}')
                        
            # Early Stopping
            if len(early_stop_score) == 0 or top1_acc > early_stop_score[-1]:
                early_stop_score.append(top1_acc)
                if len(early_stop_score) == args.early_stop_epoch:break                                      
            else: early_stop_score = list() 
        
        else:
            print(f'Epoch:{epoch_i+1}, Train Loss: {train_loss.mean():.4f}') 
        
    training_time = format_time(time.time() - t0)
    print(f'Total Training Time: {training_time}')


if __name__ == '__main__':
    args = argument_parser()
    main(args)
