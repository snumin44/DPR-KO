- **train.py** 의 하이퍼 파라미터는 다음과 같습니다.              
- 이를 참고해 쉘 스크립트를 수정해서 사용하면 됩니다. (파일 경로는 수정하지 않는 것이 좋습니다.)

|Hyper Parameter|설명|
|---|---|
|model|**베이스 모델**의 경로입니다. HuggingFace의 모델 명을 입력하면 됩니다.|
|train_data|학습에 사용하는 검증 **Train Set** 의 경로입니다. 모델을 학습하기 위해 반드시 입력해주어야 합니다.|
|valid_data|평가에 사용하는 검증 **Validation Set** 의 경로입니다. 학습 과정에서 검색 성능을 평가하기 위해 반드시 입력해주어야 합니다.|
|q_output_path|**Question Encoder**가 저장되는 경로입니다. '../pretrained_model/question_encoder'가 기본값으로 설정되어 있습니다.|
|c_output_path|**Context Encoder**가 저장되는 경로입니다.  '../pretrained_model/context_encoder'가 기본값으로 설정되어 있습니다.|
|max_length|베이스 모델 토크나이저의 **최대 토큰의 개수**입니다. BERT 모델의 최대 토큰 개수인 512 가 기본값으로 설정되어 있습니다.|
|batch_size|학습 데이터로 구성된 **배치의 크기**입니다. 배치 사이즈의 크기가 클수록 모델의 성능이 향상됩니다. 64가 기본값으로 설정되어 있습니다.|
|epochs|학습시 **epoch**의 크기입니다. 15가 기본값으로 설정되어 있습니다.|
|eval_epoch|학습시 **검색 성능 평가**를 진행하는 epoch의 단위입니다. 이 값이 1이라면 매 epoch마다 평가를 진행합니다. 1이 기본값으로 설정되어 있습니다.|
|early_stop_epoch|검색 성능이 더 이상 향상되지 않을 때 학습을 **조기 종료** 하는 epoch의 단위입니다. 이 값이 5라면 5 epoch 동안 성능 향상이 없을 때 학습이 종료됩니다. 5가 기본값으로 설정되어 있습니다.|
|pooler|모델의 출력으로 부터 임베딩을 추출할 때 사용하는 **pooler**의 종류입니다. 'pooler_output', 'cls', 'mean', 'max' 중에 선택할 수 있고, 'cls'가 기본값으로 설정되어 있습니다.|
|weight_decay|모델이 과적합되지 않도록 **가중치 감쇠 비율**을 설정합니다. 1e-2가 기본값으로 설정되어 있습니다.|
|no_decay|가중치 감쇠에서 제외할 파라미터입니다. 'bias', 'LayerNorm.weight'가 기본값으로 설정되어 있습니다.|
|temp|유사도 계산을 위한 **temperature** 입니다. 소프트맥스 함수의 분포를 조절할 때 사용됩니다. 0.05가 기본값으로 설정되어 있습니다.|
|random_seed|실행 결과를 고정하기 위한 **랜덤 시드**로, 42가 기본값으로 설정되어 있습니다.|
|dropout|모델이 과적합되지 않도록 **드롭아웃 비율**을 설정합니다. 0.1이 기본값으로 설정되어 있습니다.|
|learning_rate|모델의 **학습률**을 설정합니다. 모델이 학습할 때 가중치가 얼마나 빠르게 업데이트되는지를 결정합니다. 5e-5가 기본값으로 설정되어 있습니다.|
|eta_min|CosineAnnealingLR 스케줄러의 **최소 학습률**을 설정합니다. 0이 기본값으로 설정되어 있습니다.|
|amp|학습 시 **Automatic Mixed Precision**을 사용할지 결정합니다. 쉘 스크립트에는 사용하도록 설정되어 있습니다.|
|device|'torch.cuda.is_available()'의 값에 따라 사용할 **장치(gpu/cpu)** 를 결정합니다. 여러 대의 gpu를 사용할 경우 쉘 스크립트에서 GPU가 명시된 부분을 수정해야 합니다.|