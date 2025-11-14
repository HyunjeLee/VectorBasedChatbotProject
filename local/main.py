# main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from contextlib import asynccontextmanager

# --- 1. 설정값 (initialize_vector_db.py와 동일해야 함) ---
QDRANT_URL = "http://localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "perso_ai_qa_knowledge_base"
EMBEDDING_MODEL = "snunlp/KR-SBERT-V40K-klueNLI-augSTS" # 인덱싱에 사용한 모델과 동일

# --- 2. 전역 객체 (서버 시작 시 로드) ---
qdrant_client = None
embedding_model = None

# 응답 정확도를 위한 임계값 (0.0 ~ 1.0)
SIMILARITY_THRESHOLD = 0.75 

# --- 3. 데이터 모델 정의 ---
# 사용자 요청 시 받을 JSON 데이터 구조 (질문 텍스트)
class QueryRequest(BaseModel):
    query: str

# --- 4. 서버 시작 및 종료 이벤트 (Lifespan) ---
@asynccontextmanager
async def lifespan(_: FastAPI):
    """서버 시작 시 Qdrant 클라이언트와 임베딩 모델을 딱 한 번 로드합니다."""
    global qdrant_client, embedding_model
    try:
        qdrant_client = QdrantClient(url=QDRANT_URL, port=QDRANT_PORT)
        print("✅ Qdrant 클라이언트 연결 성공.")
        
        # 임베딩 모델 로드 (FastAPI 서버가 꺼질 때까지 메모리에 유지됨)
        embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        print("✅ 임베딩 모델 로드 성공.")
        
    except Exception as e:
        print(f"❌ 초기화 오류 발생: {e}")
        # 초기화 실패 시 서버 시작을 중단할 수 있으나, 여기서는 경고만 출력합니다.
    
    yield  # 서버 실행 중
    
    # 서버 종료 시 정리 작업이 필요하면 여기에 추가
    print("서버 종료 중...")

app = FastAPI(lifespan=lifespan)

# --- 5. 핵심 API 엔드포인트 구현 (검색 및 응답) ---
@app.post("/chat")
def chat_with_vector_db(request: QueryRequest):
    """사용자의 질문을 벡터 DB에서 검색하여 정확한 답변을 반환합니다."""
    
    user_query = request.query.strip()
    if not user_query:
        raise HTTPException(status_code=400, detail="질문을 입력해 주세요.")
        
    try:
        # 1. 질문을 벡터화 (임베딩)
        query_vector = embedding_model.encode(user_query).tolist()
        
        # 2. Qdrant에서 유사도 검색
        search_results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=1, # 가장 유사한 답변 1개만 가져옴
            with_payload=True # 저장된 답변 텍스트(payload)를 함께 가져옴
        )
        
        # 3. 검색 결과 처리 및 환각 방지 로직 (Threshold)
        if search_results and search_results[0].score >= SIMILARITY_THRESHOLD:
            # 검색된 답변의 score가 임계값을 넘으면 원본 답변을 반환
            final_answer = search_results[0].payload.get('answer', "답변 내용이 손실되었습니다.")
        else:
            # 임계값을 넘지 않거나 결과가 없으면, 안전 메시지를 반환하여 환각을 방지함
            final_answer = "죄송합니다. 현재 지식 기반(DB)에서 요청하신 정보를 찾을 수 없습니다. (유사도 낮음)"
            
        return {"query": user_query, "response": final_answer}
        
    except Exception as e:
        print(f"검색 중 예외 발생: {e}")
        raise HTTPException(status_code=500, detail="서버 오류: 벡터 검색 중 문제가 발생했습니다.")
