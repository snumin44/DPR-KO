
    
    
    
    
    
    
    
    
|Hyper Parameter|설명|
|:---:|:---:|
|model|****|    
    
    
    
    parser.add_argument('--model', type=str, required=True,
                        help='Path of validation dataset'
                       )
    parser.add_argument('--faiss_path', type=str, required=True,
                        help='Path of faiss pickle'
                       )
    parser.add_argument('--context_path', type=str, required=True,
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
