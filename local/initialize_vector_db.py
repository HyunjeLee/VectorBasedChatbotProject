# initialize_vector_db.py

import pandas as pd
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
import os

# --- 설정값 ---
# 1. DB 접속 정보: 도커에서 실행한 Qdrant의 주소와 포트
QDRANT_URL = "http://localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "perso_ai_qa_knowledge_base" # 컬렉션 이름

# 2. 임베딩 모델 선택 (한국어 문맥에 적합한 모델)
# 한국어 Sentence-BERT 모델 중 하나를 사용합니다.
EMBEDDING_MODEL = "snunlp/KR-SBERT-V40K-klueNLI-augSTS" 

# 3. 데이터 파일 경로
DATA_FILE_PATH = "sample_data.xlsx"  


def parse_qa_data(data):
    df = pd.read_excel(data, header=None, skiprows=3) # 처음 3줄은 skip
    qa_column_index = 2 
    questions = []
    answers = []

    for _, row in df.iterrows():
        value = row.iloc[qa_column_index] # 해당 칼럼의 qa_column_index 번째 셀을 가져온다.
        if not isinstance(value, str):
            continue
        text = value.strip()
        if not text:
            continue

        if text.lower().startswith("q."): # Q 또는 q로 시작하는 경우
            question = text[2:].strip() # Q. 뒤의 텍스트를 가져온다.
            if question:
                questions.append(question)
        elif text.lower().startswith("a."): # A 또는 a로 시작하는 경우
            answer = text[2:].strip() # A. 뒤의 텍스트를 가져온다.
            if answer:
                answers.append(answer)

    return questions, answers


def initialize_db():
    # 1. Qdrant 클라이언트 초기화 및 접속
    client = QdrantClient(url=QDRANT_URL, port=QDRANT_PORT)
    print(f"Qdrant 클라이언트 접속 시도: {QDRANT_URL}:{QDRANT_PORT}")

    # 2. 임베딩 모델 로드
    model = SentenceTransformer(EMBEDDING_MODEL)
    vector_dim = model.get_sentence_embedding_dimension()
    print(f"임베딩 모델 로드 완료. 벡터 차원: {vector_dim}")

    # 3. 데이터 로드 및 전처리
    questions, answers = parse_qa_data(DATA_FILE_PATH)
    print(f"로드된 Q&A 쌍 개수: {len(questions)}")
    
    # 4. Qdrant 컬렉션 생성 (권장 방식: 존재하면 삭제 후 재생성)
    
    # 4-1. 컬렉션이 존재하는지 확인하고, 존재하면 삭제합니다.
    if client.collection_exists(collection_name=COLLECTION_NAME):
        client.delete_collection(collection_name=COLLECTION_NAME)
        print(f"기존 컬렉션 '{COLLECTION_NAME}'을(를) 삭제했습니다.")
        
    # 4-2. 컬렉션을 새로 생성합니다.
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=vector_dim, 
            distance=models.Distance.COSINE
        ),
    )
    print(f"컬렉션 '{COLLECTION_NAME}' (차원: {vector_dim}, 척도: COSINE) 생성 완료.")


    # 5. 질문을 벡터화(임베딩)하고 DB에 저장 (인덱싱)
    
    # 5-1. 질문 리스트를 모델에 넣어 한 번에 벡터를 생성
    # 참고: .encode()가 벡터 변환을 자동으로 수행합니다.
    vectors = model.encode(questions).tolist() 
    print(f"벡터화 완료. {len(vectors)}개의 벡터 생성.")

    # 5-2. Qdrant에 저장할 데이터 구조 생성
    points = []
    for i, (q, a) in enumerate(zip(questions, answers)):
        # Qdrant의 PointStruct에 Q 벡터와 A 답변 텍스트를 페이로드(payload)로 저장합니다.
        points.append(
            models.PointStruct(
                id=i, # 고유 ID
                vector=vectors[i], # 질문(Q)의 벡터 데이터
                payload={
                    "question": q, 
                    "answer": a # 원본 답변(A) 텍스트를 메타데이터로 저장
                } 
            )
        )

    # 5-3. 데이터 저장 (Upsert)
    client.upsert(
        collection_name=COLLECTION_NAME,
        wait=True,
        points=points
    )
    print(f"\n✅ Qdrant DB 인덱싱 및 데이터 저장이 {len(points)}개 포인트로 완료되었습니다!")


if __name__ == "__main__":
    initialize_db()
    