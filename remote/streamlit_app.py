# streamlit_app.py

import pandas as pd
import streamlit as st
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from google import genai

# --- 1. 설정값 ---
QDRANT_URL = st.secrets["QDRANT_URL"]
QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
COLLECTION_NAME = st.secrets["QDRANT_COLLECTION"]
EMBEDDING_MODEL = "dragonkue/multilingual-e5-small-ko-v2"
SIMILARITY_THRESHOLD = 0.75  # 유사도 임계값 
# DATA_FILE_PATH = "data/Q&A.xlsx" # 로컬 파일 경로
DATA_FILE_PATH = "remote/data/Q&A.xlsx" # 원격 서버 파일 경로

print(f"--- 디버깅 정보 ---")
print(f"로드된 QDRANT_URL: {QDRANT_URL}")
print(f"로드된 QDRANT_API_KEY 길이: {len(QDRANT_API_KEY) if QDRANT_API_KEY else 0}")
print(f"로드된 COLLECTION_NAME: {COLLECTION_NAME}")
print("------------------")


# --- 2. 리소스 캐싱 (서버 시작 시 한 번만 로드) ---

def parse_qa_data(file_path):
    df = pd.read_excel(file_path, header=None, skiprows=3, engine="openpyxl") # 처음 3줄은 skip
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
            question = text[2:].lower().strip() # Q. 뒤의 텍스트를 가져온다. # lower() 전처리 과정 추가
            if question:
                questions.append(question)
        elif text.lower().startswith("a."): # A 또는 a로 시작하는 경우
            answer = text[2:].strip() # A. 뒤의 텍스트를 가져온다.
            if answer:
                answers.append(answer)

    return questions, answers

@st.cache_resource
def get_qdrant_client():
    """Qdrant 클라이언트를 로드하고 메모리에 캐싱합니다."""
    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        st.success(" ✅ Qdrant 클라이언트 연결 성공.", icon="🔗")
        return client
    except Exception as e:
        st.error(f"❌ Qdrant 연결 오류: {e}")
        return None

@st.cache_resource
def get_embedding_model():
    """임베딩 모델을 로드하고 메모리에 캐싱합니다."""
    try:
        model = SentenceTransformer(EMBEDDING_MODEL)
        st.success(" ✅ 임베딩 모델 로드 성공.", icon="🧠")
        return model
    except Exception as e:
        st.error(f"❌ 임베딩 모델 로드 오류: {e}")
        return None

@st.cache_resource
def get_gemini_client():
    """Gemini API 클라이언트를 로드하고 메모리에 캐싱합니다."""
    try:
        api_key = st.secrets.get("GEMINI_API_KEY")
        if not api_key:
            st.warning("⚠️ GEMINI_API_KEY가 설정되지 않았습니다. 질문 증강 기능을 사용할 수 없습니다.")
            return None
        client = genai.Client(api_key=api_key)
        st.success(" ✅ Gemini API 클라이언트 연결 성공.", icon="🤖")
        return client
    except Exception as e:
        st.error(f"❌ Gemini API 연결 오류: {e}")
        return None

def augment_questions_with_gemini(original_question: str, gemini_client) -> list:
    """
    Gemini API를 사용하여 주어진 질문과 유사한 질문 3개를 생성합니다.

    Args:
        original_question: 원본 질문
        gemini_client: Gemini API 클라이언트

    Returns:
        [원본 질문, 유사질문1, 유사질문2, 유사질문3] 형태의 리스트
    """
    if not gemini_client:
        # Gemini 클라이언트가 없으면 원본 질문만 반환
        return [original_question]

    try:
        prompt = f"""주어진 질문과 의미는 같지만 표현이 다른 질문 3개를 생성해주세요.
원본 질문의 핵심 의도와 맥락을 유지하면서, 다양한 표현 방식을 사용해주세요.

원본 질문: {original_question}

다음 형식으로 정확히 3개의 질문만 생성해주세요 (번호 없이, 각 질문은 한 줄로):
1. 첫 번째 유사 질문
2. 두 번째 유사 질문
3. 세 번째 유사 질문

응답 예시:
사용자 계정을 어떻게 만들 수 있나요
회원가입은 어디서 하나요
새로운 계정 생성 방법을 알려주세요"""

        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        generated_text = response.text.strip()

        # 생성된 텍스트를 줄 단위로 분리
        lines = [line.strip() for line in generated_text.split('\n') if line.strip()]

        # 번호나 불필요한 prefix 제거
        similar_questions = []
        for line in lines:
            # "1.", "2.", "3." 등의 번호 제거
            cleaned = line
            if len(line) > 0 and line[0].isdigit():
                # 숫자로 시작하면 숫자와 점, 공백 제거
                parts = line.split('.', 1)
                if len(parts) > 1:
                    cleaned = parts[1].strip()

            if cleaned and len(similar_questions) < 3:
                similar_questions.append(cleaned.lower().strip())

        # 생성된 질문이 3개 미만이면 원본 질문으로 채우기
        while len(similar_questions) < 3:
            similar_questions.append(original_question)

        # 원본 질문 + 생성된 질문 3개 반환
        return [original_question] + similar_questions[:3]

    except Exception as e:
        st.warning(f"⚠️ 질문 증강 중 오류 발생: {e}. 원본 질문만 사용합니다.")
        return [original_question]

qdrant_client = get_qdrant_client()
embedding_model = get_embedding_model()
gemini_client = get_gemini_client()

@st.cache_resource
def initialize_db():
    """Qdrant에 데이터가 없으면 초기화합니다."""

    st.info("🔄 DB 초기화 로직을 시작합니다...")

    if not qdrant_client or not embedding_model:
        st.error("서버 자원(Qdrant/모델)이 로드되지 않아 DB를 초기화할 수 없습니다.")
        return

    # 데이터 로드 및 전처리
    questions, answers = parse_qa_data(DATA_FILE_PATH)
    if not questions or not answers or len(questions) != len(answers):
        st.error("데이터 파일에서 유효한 Q&A 쌍을 찾을 수 없습니다.")
        return

    st.info(f"📚 원본 데이터: {len(questions)}개의 Q&A 쌍 로드 완료")

    # 질문 증강 수행 (Gemini API 사용)
    if gemini_client:
        st.info("🤖 Gemini를 사용하여 질문을 증강하는 중...")
        augmented_questions = []
        augmented_answers = []

        # 진행 상황 표시
        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, (question, answer) in enumerate(zip(questions, answers)):
            # 진행 상황 업데이트
            progress = (idx + 1) / len(questions)
            progress_bar.progress(progress)
            status_text.text(f"처리 중: {idx + 1}/{len(questions)} ({progress*100:.1f}%)")

            # 질문 증강: 원본 질문 + 유사 질문 3개 = 총 4개
            similar_questions = augment_questions_with_gemini(question, gemini_client)

            # 각 질문을 같은 답변과 매핑
            for aug_question in similar_questions:
                augmented_questions.append(aug_question)
                augmented_answers.append(answer)
                st.info(f"증강된 질문: {aug_question}  =>  답변: {answer}")

        progress_bar.empty()
        status_text.empty()

        questions = augmented_questions
        answers = augmented_answers
        st.success(f"✅ 질문 증강 완료: {len(questions)}개의 질문으로 확장되었습니다.", icon="🚀")
    else:
        st.warning("⚠️ Gemini API를 사용할 수 없어 질문 증강을 건너뜁니다.")

    # 컬렉션 생성
    vector_dim = embedding_model.get_sentence_embedding_dimension()
    if qdrant_client.collection_exists(collection_name=COLLECTION_NAME):
        qdrant_client.delete_collection(collection_name=COLLECTION_NAME)
    print(f"기존 컬렉션 '{COLLECTION_NAME}'을(를) 삭제했습니다.")

    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=vector_dim,
            distance=models.Distance.COSINE
        ),
    )
    print(f"컬렉션 '{COLLECTION_NAME}' 생성 완료.")
    st.info(f"컬렉션 '{COLLECTION_NAME}'이(가) 생성되었습니다.")

    # 데이터 임베딩 및 업로드
    st.info("🧠 질문을 임베딩하는 중...")
    embeddings = embedding_model.encode(questions).tolist()
    points = [
        models.PointStruct(
            id=i,
            vector=embeddings[i],
            payload={"question": questions[i], "answer": answers[i]}
        )
        for i in range(len(questions))
    ]
    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )
    st.success(f"✅ {len(points)}개의 Q&A 데이터가 업로드되었습니다.", icon="📥")

    return True

initialize_db()  # 앱 시작 시 DB 초기화 시도

# --- 3. 핵심 검색 함수 (FastAPI의 /chat 역할) ---

def search_qdrant_db(user_query: str):
    """
    사용자의 질문을 벡터화하여 Qdrant에서 검색하는 함수입니다.
    FastAPI의 POST /chat 엔드포인트 로직과 동일합니다.
    """
    if not qdrant_client or not embedding_model:
        st.error("서버 자원(Qdrant/모델)이 로드되지 않아 검색할 수 없습니다.")
        return "죄송합니다. 서버 초기화에 실패했습니다."
    
    user_query = user_query.lower().strip() # lower() 전처리 과정 추가 # trim() 전처리 과정 추가

    # 1. 질문을 벡터화 (임베딩)
    with st.spinner("질문을 분석하고 지식 기반을 검색 중..."):
        query_vector = embedding_model.encode(user_query).tolist()
        
        # 2. Qdrant에서 유사도 검색
        search_results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=1,
            with_payload=True
        )
    
    # 3. 검색 결과 처리 및 환각 방지 로직 (Threshold)
    if search_results and search_results[0].score >= SIMILARITY_THRESHOLD:
        final_answer = search_results[0].payload.get('answer', "답변 내용이 손실되었습니다.")
        st.info(f"유사도 점수: {search_results[0].score:.4f}")
    else:
        final_answer = "죄송합니다. 현재 지식 기반(DB)에서 요청하신 정보를 찾을 수 없습니다. (유사도 낮음)"
        if search_results:
             st.warning(f"유사도 점수가 임계값({SIMILARITY_THRESHOLD})보다 낮습니다: {search_results[0].score:.4f}")
            
    return final_answer


# --- 4. Streamlit UI 구성 ---    

st.title("🤖 Qdrant 기반 Q&A 챗봇")

# Session State 초기화: 채팅 기록을 저장
if 'messages' not in st.session_state:
    st.session_state.messages = []

# 이전 채팅 기록 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력 처리 (버튼 클릭 대신 채팅 입력 방식 사용)
if prompt := st.chat_input("질문을 입력하세요"):
    # 사용자 질문을 기록
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # 검색 로직 실행 (FastAPI의 POST 요청을 대체)
    response = search_qdrant_db(prompt)

    # 챗봇 응답을 기록 및 표시
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)