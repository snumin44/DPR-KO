import datetime

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed))) # Round to the nearest second.
    return str(datetime.timedelta(seconds=elapsed_rounded)) # Format as hh:mm:ss

def get_topk_accuracy(faiss_index, answer_idx, positive_idx): 

    top1_correct = 0
    top5_correct = 0
    top10_correct = 0
    top20_correct = 0
    top50_correct = 0
    top100_correct = 0
    
    for idx, answer in enumerate(answer_idx):
        
        #  *** faiss index: (question num * k) ***
        #      [[73587  2746 15265 96434 ...]
        #       [98388 13550 93912 92610 ...]
        #                    ...
        #       [97530 93498 16607 98168 ...]
        #       [52308 24908 70869 20824 ...]
        #       [44597 35140  7572  4596 ...]]
         
        retrieved_idx = faiss_index[idx] 
            
        retrieved_idx = [positive_idx[jdx] for jdx in retrieved_idx]   
        
        if any(ridx in answer for ridx in retrieved_idx[:1]):
            top1_correct += 1
        if any(ridx in answer for ridx in retrieved_idx[:5]):
            top5_correct += 1
        if any(ridx in answer for ridx in retrieved_idx[:10]):
            top10_correct += 1
        if any(ridx in answer for ridx in retrieved_idx[:20]):
            top20_correct += 1
        if any(ridx in answer for ridx in retrieved_idx[:50]):
            top50_correct += 1
        if any(ridx in answer for ridx in retrieved_idx[:100]):
            top100_correct += 1
        
    top1_accuracy = top1_correct / len(answer_idx)
    top5_accuracy = top5_correct / len(answer_idx)
    top10_accuracy = top10_correct / len(answer_idx)    
    top20_accuracy = top20_correct / len(answer_idx)
    top50_accuracy = top50_correct / len(answer_idx)
    top100_accuracy = top100_correct / len(answer_idx)

    return {
        'top1_accuracy':top1_accuracy,
        'top5_accuracy':top5_accuracy,
        'top10_accuracy':top10_accuracy,        
        'top20_accuracy':top20_accuracy,
        'top50_accuracy':top50_accuracy,
        'top100_accuracy':top100_accuracy,
    }