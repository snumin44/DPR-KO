# 🍊 DPR(Dense Passage Retrieval)-KO

- **한국어 DPR(Dense Passage Retrieval) 모델**을 학습하는 코드입니다.
- 한국어 위키피디아 덤프를 이용해 모델의 검색 성능을 평가할 수 있습니다.
- [Facebook의 DPR 코드](https://github.com/facebookresearch/DPR)와는 다른 구성입니다. 몇 가지 문제점을 해결하기 위해 새롭게 코드를 작성했습니다.      

## 1. Dense Passage Retrieval


<img src="images/dpr_structure.PNG" width="400" height="240" alt="DPR">

- DPR은 Facebook에서 공개한 **Dense Vector 기반 검색 모델**(또는 방법론)입니다.
- 질문을 인코딩하는 **Question Encoder**와 텍스트를 인코딩하는 **Context Encoder**로 이루어집니다.
- 서로 대응하는 질문과 텍스트 사이의 유사도는 키우고, 그렇지 않은 질문과 텍스트의 유사도는 낮추는 방식으로 학습합니다.    


## 2. DPR-KO

## 3. Implementation

## 4. Performance
