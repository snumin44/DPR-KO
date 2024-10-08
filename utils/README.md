- **rank_bm25** 라이브러리의 특성상 학습 데이터가 평가 코퍼스와 일치해야 Reranking이 가능합니다.
- 이 코드의 BM25 모델은 **띄어쓰기** 단위로 토크나이징된 텍스트를 학습합니다.
- **train_bm25.py** 의 하이퍼 파라미터는 다음과 같습니다. 
- 이를 참고해 쉘 스크립트를 수정해서 사용하면 됩니다. (파일 경로는 수정하지 않는 것이 좋습니다.)
    
|Hyper Parameter|설명|
|---|---|
|bm25_corpus|BM25 모델 학습에 사용하는 **코퍼스의 유형**입니다. 'all'은 위키피디아 덤프와 Validation Set의 텍스트를 학습 코퍼스로 사용하고, 'wiki'는 위키피디아 덤프만 학습 코퍼스로 사용합니다. rank_bm25 라이브러리의 특성상 학습 데이터가 평가 코퍼스와 일치해야 Reranking에 사용할 수 있습니다. 'wiki'가 기본값으로 설정되어 있습니다.|
|valid_data|**Validation Set**의 경로입니다. bm25_corpus로 'all'을 선택했다면 반드시 입력해주어야 합니다.|
|wiki_path|**위키피디아 덤프**의 경로입니다. '../wikidump/text'가 기본값으로 설정되어 있습니다.|
|save_path|**BM25 모델**이 담긴 pickle 파일이 저장되는 경로입니다. default='../pickles'가 기본값으로 설정되어 있습니다.|
|num_sent|위키피디아 덤프를 chunk로 분할 할 때 하나의 chunk에 포함된 **문장의 수**입니다. 5가 기본값으로 설정되어 있습니다.|
|overlap|연속된 chunk 간에 서로 **겹치는 문장의 수**입니다. 0이 기본값으로 설정되어 있습니다.|
|cpu_workers|	위키피디아 덤프를 chunk로 분할하는 과정은 Multi-processing 으로 이루어지는데, 이때 사용할 **cpu 코어의 개수**입니다. 쉘 스크립트에서 해당 부분을 제거하면 자동으로 모든 cpu 코어를 사용합니다. 많은 cpu 코어를 이용할 수록 위키피디아 덤프를 빠르게 분할할 수 있습니다.|
