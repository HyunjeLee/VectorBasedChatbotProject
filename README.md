# VectorBasedChatbotProject

벡터 기반 Semantic 검색을 활용한 한국어 Q&A 챗봇 프로젝트

## 프로젝트 개요

이 프로젝트는 **Qdrant 벡터 데이터베이스**와 **한국어 임베딩 모델**을 활용하여 사용자 질문에 대한 정확한 답변을 제공하는 검색 기반 챗봇입니다. LLM을 사용하지 않고 순수 벡터 유사도 검색만으로 동작하며, Gemini API를 통한 번역 기능으로 다국어 지원 및 검색 정확도를 향상시킵니다.

## 주요 특징

- **다중 벡터 검색**: 한글/영어 질문과 답변을 모두 벡터화하여 검색 정확도 극대화
- **이중 검색 메커니즘**: 사용자 질문을 한글과 영어로 모두 검색하여 일치하는 답변 선택
- **질문/답변 증강**: Gemini API를 활용한 자동 번역으로 데이터 4배 확장
- **환각 방지**: 유사도 임계값(0.75)을 통한 신뢰성 있는 답변만 제공
- **실시간 UI**: Streamlit 기반의 직관적인 채팅 인터페이스

---

## 기술 스택

### 프론트엔드 & 백엔드
- **Streamlit**: 웹 UI 프레임워크
  - Streamlit Cloud를 통한 배포
- **Python 3.x**: 메인 개발 언어

### 벡터 데이터베이스
- **Qdrant**: 고성능 벡터 검색 엔진
  - Qdrant Cloud를 통한 배포
  - COSINE 거리 메트릭 사용

### 임베딩 & AI
- **Sentence Transformers**: 텍스트 임베딩 생성
  - 시도 모델 : 
    
    `snunlp/KR-SBERT-V40K-klueNLI-augSTS` -> 마지막 업데이트 : 3년 전
    
    `jinaai/jina-embeddings-v3` -> mteb 기준 상위 모델, zero-shot 100% -> 리소스 1GB 초과로 Streamlit Cloud 환경에서 실행 불가
  
  - 현재 모델: 

    `dragonkue/multilingual-e5-small-ko-v2` (한국어 특화)
    - 다국어 지원 텍스트 임베딩 모델 중 실행 조건(리소스 크기)을 만족
    - 한국어로 Fine-tuning된 버전 존재

    
- **Google Gemini API**: 자동 번역
  - 모델: `gemini-2.0-flash`
  - 질문/답변 영어 번역

### 데이터 처리
- **Pandas**: Excel 데이터 파싱
- **OpenPyXL**: .xlsx 파일 읽기

---

## 벡터 DB 및 임베딩 방식

### 1. 임베딩 모델
```python
EMBEDDING_MODEL = "dragonkue/multilingual-e5-small-ko-v2"
```
- **한국어 최적화**: 한국어 텍스트에 대한 높은 정확도
- **다국어 지원**: 한글-영어 간 의미적 유사성 보존
- **벡터 차원**: 모델 기본 차원 사용

### 2. 데이터 구조

#### 원본 데이터
- Excel 파일 (Q&A.xlsx)에서 Q&A 쌍 로드
- 예: 13개의 Q&A 쌍

#### 데이터 증강 프로세스
각 Q&A 쌍에 대해 **4개의 벡터 포인트** 생성:

1. **한글 질문** → 임베딩
2. **영어 질문** → Gemini 번역 → 임베딩
3. **한글 답변** → 임베딩
4. **영어 답변** → Gemini 번역 → 임베딩

**결과**: 13개 Q&A → **최대 52개 벡터 포인트**

#### Payload 구조
모든 벡터 포인트는 동일한 payload 구조를 가짐:
```json
{
  "question": "원본 한글 질문",
  "answer": "원본 한글 답변"
}
```

### 3. 벡터 검색 설정
```python
vectors_config=models.VectorParams(
    size=vector_dim,
    distance=models.Distance.COSINE  # 코사인 유사도
)
```

---

## 정확도 향상 전략

### 0. 질문 증강 (DB 데이터셋 다양화)

**전략**: AI에게 질문과 답변을 전달해 비슷한 질문을 3개 추가로 생성 -> 1개의 답변에 해당하는 총 4개의 질문 벡터화

**효과**:
- 사용자 질문에 대해 높은 확률로 DB 기반 답변 출력
- 답변의 정확도가 낮아짐 (생성된 질문의 품질, 기존 질문과 중복된 질문 생성)

**-> 해당 기능 삭제**

### 1. 다중 벡터 증강 (Multi-Vector Augmentation)

**전략**: 질문과 답변을 모두 벡터화
```
원본: 13개 Q&A
증강: 52개 벡터 포인트
  ├─ 한글 질문 13개
  ├─ 영어 질문 13개
  ├─ 한글 답변 13개
  └─ 영어 답변 13개
```

**효과**:
- 사용자가 질문 형태가 아닌 답변 키워드로 질문해도 검색 가능
- 예: "API 키를 어디서 받나요?" → 답변 벡터 "API 키는 대시보드에서..."와도 매칭

### 2. 이중 언어 검색 (Dual-Language Search)

**프로세스**:
```
사용자 질문 입력
    │
    ├─→ [한글 검색] Top-3 추출
    │       │
    │       └─→ 결과 A
    │
    └─→ [영어 번역] → [영어 검색] Top-3 추출
            │
            └─→ 결과 B
                │
                ├─→ A ∩ B (일치하는 답변)
                │     └─→ 최고 유사도 선택
                │
                └─→ A (일치 없음)
                      └─→ 한글 1위 반환
```

**효과**:
- 한글/영어 양쪽에서 일치하는 답변 = 높은 신뢰도
- 교차 검증을 통한 정확도 향상

### 3. 유사도 임계값 필터링

```python
SIMILARITY_THRESHOLD = 0.75  # 75% 이상만 답변
```

**효과**:
- 낮은 유사도 결과 필터링 → 환각 방지
- "답변을 찾을 수 없습니다" 명시적 응답

### 4. 전처리 정규화

```python
user_query = user_query.lower().strip()  # 소문자 변환 + 공백 제거
question = text[2:].lower().strip()      # Q. 제거 + 정규화
```

**효과**:
- 대소문자, 공백 차이로 인한 오차 제거
- 일관된 벡터 임베딩

### 5. 실시간 번역 검증

```python
if english_question:
    # 번역 성공 → 영어 벡터 추가
else:
    # 번역 실패 → 한글만 사용
```

**효과**:
- 번역 실패 시 안전하게 원본만 사용
- 데이터 품질 유지

---

## 프로젝트 구조

```
VectorBasedChatbotProject/
├── local/                      # 로컬 개발 환경
│   ├── main.py                # FastAPI 백엔드 (localhost)
│   ├── streamlit_app.py       # Streamlit UI (localhost)
│   └── initialize_vector_db.py # DB 초기화 (Docker)
├── remote/                     # 원격 배포 환경
│   ├── streamlit_app.py       # 통합 앱 (UI + 백엔드)
│   └── data/Q&A.xlsx          # Q&A 데이터
├── requirements.txt           # 의존성 목록
└── README.md                  # 프로젝트 문서
```

---

## 핵심 기능 상세

### 질문 검색 프로세스 (TOP-3 교차 검증)

```python
def search_qdrant_db(user_query: str):
    # 1. 한글 검색 (Top-3)
    korean_results = qdrant_client.search(query_vector, limit=3)

    # 2. 영어 번역 후 검색 (Top-3)
    english_query = translate_to_english(user_query)
    english_results = qdrant_client.search(english_vector, limit=3)

    # 3. 일치하는 답변 찾기
    matching = find_common_answers(korean_results, english_results)

    # 4. 최고 유사도 선택
    if matching:
        return max(matching, key=lambda x: x.score)
    else:
        return korean_results[0]  # 한글 1위
```

### 데이터 증강 프로세스

```python
for question, answer in qa_pairs:
    # 1. 한글 질문
    vectors.append(embed(question))

    # 2. 영어 질문
    en_q = translate(question)
    vectors.append(embed(en_q))

    # 3. 한글 답변
    vectors.append(embed(answer))

    # 4. 영어 답변
    en_a = translate(answer)
    vectors.append(embed(en_a))
```

---

## 향후 개선 방향

- [ ] BM25 하이브리드(키워드 + 벡터) 검색 추가
- [ ] Reranker를 사용한 정밀 검색