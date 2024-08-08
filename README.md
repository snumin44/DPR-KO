# 🍊 DPR(Dense Passage Retrieval)-KO

- **한국어 DPR(Dense Passage Retrieval) 모델**을 학습하는 코드입니다.
- 한국어 위키피디아 덤프를 이용해 모델의 검색 성능을 평가할 수 있습니다.
- [Facebook의 DPR 코드](https://github.com/facebookresearch/DPR)와는 다른 구성입니다. 몇 가지 문제점을 해결하기 위해 새롭게 코드를 작성했습니다.      

## 1. Dense Passage Retrieval

&nbsp; <img src="images/dpr_structure.PNG" width="400" height="240" alt="DPR">

- DPR은 Facebook에서 공개한 **Dense Vector 기반 검색 모델**(또는 방법론)입니다.
- 질문을 인코딩하는 **Question Encoder**와 텍스트를 인코딩하는 **Context Encoder**로 이루어집니다.
               
&nbsp;&nbsp; <img src="images/dpr_loss.PNG" width="380" height="180" alt="DPR">

- 서로 대응하는 질문과 텍스트의 유사도는 키우고, 그렇지 않은 질문과 텍스트의 유사도는 낮추는 방식으로 학습합니다.
  - Batch Size가 3일 때, 두 인코더의 임베딩으로 (3 x 768) * (768 x 3) = (3 x 3)의 **Similarity Matrix**를 만들 수 있습니다. 이것의 **주대각선**이 서로 대응하는 질문과 텍스트의 유사도에 해당합니다.
  - Similarity Matrix와 **주대각선의 값이 1인 레이블**의 Cross Entropy를 줄여, 서로 대응하는 질문과 텍스트의 유사도가 1에 가까워지도록 합니다. 레이블의 다른 값은 0이므로 서로 대응하지 않는 질문과 텍스트의 유사도는 0에 가까워집니다(in-batch negative).
- 이상의 방법으로 학습된 두 인코더를 이용해 **질문과 유사도가 가장 큰 텍스트**를 추출할 수 있습니다.               

## 2. DPR-KO



## 3. Implementation

## 4. Performance
