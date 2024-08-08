# 🍊 DPR(Dense Passage Retrieval)-KO

- **한국어 DPR(Dense Passage Retrieval) 모델**을 학습하는 코드입니다.
- 한국어 위키피디아 덤프를 이용해 모델의 검색 성능을 평가할 수 있습니다.
- [Facebook의 DPR 코드](https://github.com/facebookresearch/DPR)와는 다른 구성입니다. 몇 가지 문제점을 해결하기 위해 새롭게 코드를 작성했습니다.      

## 1. Dense Passage Retrieval

&nbsp; <img src="images/dpr_structure.PNG" width="400" height="240" alt="DPR">

- DPR은 Facebook에서 공개한 **Dense Vector 기반 검색 모델**(또는 방법론)입니다.
- 질문을 인코딩하는 **Question Encoder**와 텍스트를 인코딩하는 **Context Encoder**로 이루어집니다.
               
&nbsp;&nbsp; <img src="images/dpr_loss.PNG" width="380" height="180" alt="DPR">

- 상응하는 질문과 텍스트의 유사도는 키우고, 그렇지 않은 질문과 텍스트의 유사도는 낮추는 방식으로 학습합니다.
  - Batch Size가 3일 때, 두 인코더의 임베딩으로 (3 x 768) * (768 x 3) = (3 x 3)의 **Similarity Matrix**를 만들 수 있습니다. 이것의 **주대각선**이 서로 대응하는 질문과 텍스트의 유사도에 해당합니다.
  - Similarity Matrix와 **주대각선의 값이 1인 레이블**의 Cross Entropy를 줄여, 서로 대응하는 질문과 텍스트의 유사도가 1에 가까워지도록 합니다. 레이블의 다른 값은 0이므로, 서로 대응하지 않는 질문과 텍스트의 유사도는 0에 가까워집니다(in-batch negative).
- 이상의 방법으로 학습된 두 인코더를 이용해 **질문과 유사도가 가장 큰 텍스트**를 추출할 수 있습니다.               

## 2. DPR-KO
- **DPR-KO**는 위의 구조와 학습 방법론을 차용했으나 전혀 다른 코드로 이루어져 있습니다.
- 특히 다음과 같은 부분에서 Facebook이 공개한 기존의 DPR과 다릅니다. 

#### A. 인덱스 기반 평가 방식
- 기존의 DPR은 검색한 텍스트가 **'정답(answer)'** 을 포함하고 있으면 올바른 텍스트를 추출한 것으로 간주합니다.
  - "2024년 올림픽이 열린 도시는?"이란 질문에 대해 "파리"라는 단어가 있는 텍스트는 올바른 텍스트로 간주됩니다.
  - 따라서 **단답형 정답이 없는 데이터 셋**은 기존의 DPR 코드로 학습하기 어렵습니다.

```
# DPR-KO 학습/평가 데이터 예시


```
- 이 문제를 해결하기 위해 **'Gold Passage'** 를 찾는 방식으로 평가 방식을 수정했습니다.
  - 위 예시에서처럼 질문에 대응하는 텍스트(positive_ctx)는 **모두 고유한 인덱스**를 갖습니다.
  - 평가시 질문의 **정답 인덱스 리스트(answer_idx)** 에 포함된 인덱스를 지니는 것만 올바른 텍스트로 간주됩니다.
  - 학습 과정에서의 검색 성능 평가는 **Validation Set의 질문과 텍스트**만으로 이루어집니다.
  - 따라서 정답이 없는 **'질문-텍스트'** 구성의 데이터 셋으로도 DPR 모델 학습이 가능합니다.
- 위키 피디아 덤프를 이용한 검색 성능 평가는 **'위키피디아 덤프 + Valiation Set'** 환경에서 이루어집니다.
  - 위키 피디아 덤프의 텍스트는 Validation Set의 마지막 인덱스보다 큰 수를 차례로 인덱스로 부여 받습니다.
  - 위키 피디아 덤프를 인코딩 하는 과정에서 Validation Set의 제목과 일치하는 텍스트는 삭제됩니다.
  - 이후 Validation Set의 질문이 주어지면 전체 코퍼스에서 대응하는 인덱스를 지니는 텍스트를 추출하게 됩니다.

  
  
#### (2) Hard Negative 구현
 



## 3. Implementation

## 4. Performance
