- **evaluate_retrieval.py** 의 하이퍼 파라미터는 다음과 같습니다.              
- 이를 참고해 쉘 스크립트를 수정해서 사용하면 됩니다. (파일 경로는 수정하지 않는 것이 좋습니다.)

|Hyper Parameter|설명|
|---|---|
|model|질문을 임베딩으로 변환하는 **Question Encoder** 의 경로입니다. 모델이 저장된 로컬 경로나 HuggingFace의 모델 명을 입력하면 됩니다.|
|valid_data|평가에 사용하는 검증 **Validation Set** 의 경로입니다. 검색 성능을 평가하기 위해서는 반드시 입력해주어야 합니다.|
|faiss_path|**Faiss Index**가 담긴 pickle 파일의 경로입니다. 검색 성능을 평가하기 위해서는 반드시 입력해주어야 합니다.|
|bm25_path|**BM25 모델**이 담긴 pickle 파일의 경로입니다. 검색 과정에 BM25를 사용한 Reranking을 추가하고 싶을 때 입력합니다.|
|train_bm25|임베딩을 만들 때 사용했던 동일한 텍스트 집합으로 **BM25 모델**을 학습합니다. rank_bm25 라이브러리의 특성으로 인해 임베딩을 만들 때 사용했던 텍스트 집합과 동일한 집합으로 학습해야 BM25모델을 Reranking에 사용할 수 있습니다.|
|faiss_weight|BM25를 이용한 Reranking 과정에서 **Dense Vector에 기반한 검색 점수**에 부여되는 가중치입니다. 기본값은 1로 설정되어 있습니다.|
|bm25_weight|BM25를 이용한 Reranking 과정에서 **BM25 모델의 점수**에 부여되는 가중치입니다. 기본값은 0.5로 설정되어 있습니다.|
|search_k|DPR 모델이 **검색하는 문서의 개수**입니다. Reranking은 이 문서 내에서 이루어집니다. 기본값은 2,000으로 설정되어 있습니다.|
|max_length|Question Encoder 토크나이저의 **최대 토큰의 개수**입니다. BERT 모델의 최대 토큰 개수인 512 가 기본값으로 설정되어 있습니다.|
|batch_size|Question Encoder에 전달하는 데이터로 구성된 **배치의 크기**입니다. 64가 기본값으로 설정되어 있습니다.|
|device|'torch.cuda.is_available()'의 값에 따라 사용할 **장치(gpu/cpu)** 를 결정합니다. 여러 대의 gpu를 사용할 경우 쉘 스크립트에서 GPU가 명시된 부분을 수정해야 합니다.|
