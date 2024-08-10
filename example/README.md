- Semantic Search를 위해 다음과 같은 파일이 필요합니다.
  - **Question Encoder**
  - **faiss_pickle.pkl**
  - **context_pickle.pkl**
  - **(bm25_pickle.pkl)**
- **run_semantic_search.sh**의 하이퍼 파라미터는 다음과 같습니다.
- 이를 참고해 쉘 스크립트를 수정해서 사용하면 됩니다. (파일 경로는 수정하지 않는 것이 좋습니다.)

|Hyper Parameter|설명|
|:---:|:---:|
|model|질문을 임베딩으로 변환하는 **Question Encoder** 의 경로입니다. 모델이 저장된 로컬 경로나 HuggingFace의 모델 명을 입력하면 됩니다. 벡터 DB 구축에 사용된 Context Encoder와 함께 학습된 모델이어야 합니다.
|faiss_path|**Faiss Index**가 담긴 pickle 파일의 경로입니다. 검색 성능을 평가하기 위해서는 반드시 입력해주어야 합니다.|
|context_path|**제목과 텍스트**가 담긴 pickle 파일의 경로입니다. 검색 성능을 평가하기 위해서는 반드시 입력해주어야 합니다.|
|bm25_path|**BM25 모델**이 담긴 pickle 파일의 경로입니다. 검색 과정에 BM25를 사용한 Reranking을 추가하고 싶을 때 입력합니다.|
|pooler|모델의 출력으로 부터 임베딩을 추출할 때 사용하는 **pooler**의 종류입니다. 'pooler_output', 'cls', 'mean', 'max' 중에 선택할 수 있고, 'cls'가 기본값으로 설정되어 있습니다.|
|faiss_weight|BM25를 이용한 Reranking 과정에서 **Dense Vector**에 기반한 검색 점수에 부여되는 가중치입니다. 기본값은 1로 설정되어 있습니다.|
|bm25_weight|BM25를 이용한 Reranking 과정에서 **BM25 모델**의 점수에 부여되는 가중치입니다. 기본값은 0.5로 설정되어 있습니다.|
|search_k|DPR 모델이 **검색**하는 문서의 개수입니다. Reranking은 이 문서 내에서 이루어집니다. 기본값은 2,000으로 설정되어 있습니다.|
|return_k|시스템이 최정적으로 **출력**하는 검색 문서의 개수입니다. 기본값은 5로 설정되어 있습니다.|
|max_length|Question Encoder 토크나이저의 **최대 토큰의 개수**입니다. BERT 모델의 최대 토큰 개수인 512 가 기본값으로 설정되어 있습니다.|
|device|'torch.cuda.is_available()'의 값에 따라 사용할 **장치(gpu/cpu)** 를 결정합니다. 여러 대의 gpu를 사용할 경우 쉘 스크립트에서 GPU가 명시된 부분을 수정해야 합니다.|
