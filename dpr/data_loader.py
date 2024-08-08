import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoConfig

class BiEncoderDataset(Dataset):
    def __init__(self, question, positive_ctx, answer_idx=None, hard_neg_ctx=None):
        super(BiEncoderDataset, self).__init__()
        self.question = question
        self.answer_idx = answer_idx
        self.positive_ctx = positive_ctx
        self.hard_neg_ctx = hard_neg_ctx
        
    @classmethod
    def load_train_dataset(cls, file_path):
        with open(file_path) as infile:
            dataset = json.load(infile)
            
        question, positive_ctx, hard_neg_ctx = [], [], []
        
        for idx, sample in enumerate(dataset):
            p = sample['positive']
            q = sample['question']

            positive_ctx.extend(p)           
            question.extend([q] * len(p))
            
            if 'hard_neg' in sample.keys():                
                if len(p) == len(sample['hard_neg']):
                    hard_neg_ctx.extend(sample['hard_neg'])
                                               
        if hard_neg_ctx and len(hard_neg_ctx) == len(positive_ctx):
            return cls(question, positive_ctx, hard_neg_ctx=hard_neg_ctx)
        else:
            if hard_neg_ctx:
                print(f"The number of hard negatives ({len(hard_neg_ctx)}) does not match the number of positives ({len(positive_ctx)}). Hard negatives were not loaded.")
        
        return cls(question, positive_ctx)

    @classmethod
    def load_valid_dataset(cls, file_path):
        with open(file_path) as infile:
            dataset = json.load(infile) 

        question, answer_idx, positive_ctx = [], [], []
        
        for idx, sample in enumerate(dataset):
            p = sample['positive']
            q = sample['question']
            a = sample['answer_idx']

            positive_ctx.extend(p)
            answer_idx.extend([a])            
            question.extend([q])

        return cls(question, positive_ctx, answer_idx=answer_idx)
                
    def __len__(self):
        return len(self.question)
    
    def __getitem__(self, index):
        if self.hard_neg_ctx: 
            return {'question':self.question[index],
                    'positive_ctx':self.positive_ctx[index],
                    'hard_neg_ctx':self.hard_neg_ctx[index],}
        else:
            return {'question':self.question[index],
                    'positive_ctx':self.positive_ctx[index],}


class DataCollator(object):

    def __init__(self, args):
        self.config = AutoConfig.from_pretrained(args.model)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model, config=self.config)
               
        self.padding = args.padding         
        self.max_length = args.max_length   
        self.truncation = args.truncation   

    def __call__(self, samples):
        question = []        
        positive_idx, positive_txt, positive_title = [], [], []        
        hard_neg_idx, hard_neg_txt, hard_neg_title = [], [], []

        for sample in samples:
            question.append(sample['question'])
            positive_idx.append(sample['positive_ctx']['idx'])
            positive_txt.append(sample['positive_ctx']['text'])
            positive_title.append(sample['positive_ctx']['title'])
                        
            if 'hard_neg_ctx' in sample.keys():
                hard_neg_idx.append(sample['hard_neg_ctx']['idx'])
                hard_neg_txt.append(sample['hard_neg_ctx']['text'])
                hard_neg_title.append(sample['hard_neg_ctx']['title'])
                            
        pos_num = len(samples)
        
        total_txt = positive_txt + hard_neg_txt
        total_title = positive_title + hard_neg_title 
    
        question_encode = self.tokenizer(
            question,
            padding = self.padding,
            truncation = self.truncation,
            max_length = self.max_length
            )
        
        context_encode = self.tokenizer(
            total_title,
            total_txt,
            padding = self.padding,
            truncation = self.truncation,
            max_length = self.max_length
            )        
        
        # question_batch: (batch size, sequence_length)
        question_batch = {
            'input_ids':torch.tensor(question_encode['input_ids']),
            'attention_mask':torch.tensor(question_encode['attention_mask']),
            'token_type_ids':torch.tensor(question_encode['token_type_ids'])
        }
          
        features = {}
        for key in context_encode:
            features[key] = [[context_encode[key][i],
                             context_encode[key][i+pos_num]] 
                             if len(hard_neg_txt) > 0 else [context_encode[key][i]] for i in range(pos_num)] 
            
        # context_batch : (batch size, 2 or 1, sequence_length)
        context_batch = {
            'input_ids':torch.tensor(features['input_ids']),
            'attention_mask':torch.tensor(features['attention_mask']),
            'token_type_ids':torch.tensor(features['token_type_ids']),
            }
        
        return question_batch, context_batch