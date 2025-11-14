import streamlit as st
from streamlit.elements.lib.layout_utils import Height
import requests

# session_state 초기화
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None

# 서버 설정
CHAT_SERVER_URL = "http://localhost:8000/chat"
REQUEST_TIMEOUT = 30  # 30초 타임아웃


# AI 응답 생성 함수 (FastAPI 서버 연동)
def generate_ai_response(user_question):
    """사용자 질문을 FastAPI 서버로 전송하고 응답을 받아옴"""
    try:
        # FastAPI 서버로 POST 요청 전송
        response = requests.post(
            CHAT_SERVER_URL,
            json={"query": user_question},
            timeout=REQUEST_TIMEOUT
        )

        # 응답 상태 코드 확인
        response.raise_for_status()

        # JSON 응답 파싱
        result = response.json()
        return result.get("response", "응답을 받을 수 없습니다.")

    except requests.exceptions.ConnectionError:
        return "❌ 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인해주세요. (http://localhost:8000)"
    except requests.exceptions.Timeout:
        return "⏱️ 서버 응답 시간이 초과되었습니다. 다시 시도해주세요."
    except requests.exceptions.HTTPError as e:
        return f"❌ 서버 오류가 발생했습니다: {e}"
    except Exception as e:
        return f"❌ 예상치 못한 오류가 발생했습니다: {e}"


# 1. 페이지 설정: 와이드 레이아웃 사용 및 사이드바 초기 상태를 'expanded'로 설정
st.set_page_config(
    layout="wide", initial_sidebar_state="expanded", page_title="Custom Chat UI"
)

# 2. 사용자 지정 CSS 적용
# - 왼쪽 사이드바 상단의 Streamlit 메뉴와 "Plus 사용하기" 버튼(Upgrade) 숨기기
# - "내 채팅" 목록 영역을 비우기 (또는 CSS로 특정 요소를 숨김)
# - 상단 deploy 버튼 숨기기 (모든 화면에서 작동)

custom_css = """
<style>

/* 왼쪽 상단의 기본 햄버거 메뉴를 '아이콘을 클릭해 열고 닫을 수 있는' 기능으로 사용합니다.
   Streamlit의 기본 사이드바 토글 버튼이 이 역할을 합니다.
   이 버튼의 가시성은 기본적으로 유지됩니다. */
div[data-testid="collapsedControl"] {
    visibility: visible;
}

/* 좌우 패딩 */
.block-container {
    padding-left: 200px;
    padding-right: 200px;
}

/* 사용자 메시지 스타일 - 라이트모드: 연한 보라색, 다크모드: 진한 보라/남색 */
.user-message {
    background-color: #f3e5f5;
    border-radius: 15px;
    padding: 10px 20px;
    margin: 10px 0;
    margin-left: auto;
    margin-right: 0;
    display: block;
    max-width: 80%;
    word-wrap: break-word;
    white-space: pre-wrap;
    text-align: left;
}

/* 다크모드일 때 사용자 메시지 배경색 변경 */
@media (prefers-color-scheme: dark) {
    .user-message {
        background-color: #4a3a5c;
    }
}

.user-message-container {
    display: flex;
    justify-content: flex-end;
    margin: 10px 0;
}

/* AI 응답은 왼쪽 정렬 */
.ai-message {
    margin: 10px 0;
    padding: 10px 0;
    line-height: 1.6;
    text-align: left;
}

.ai-message-container {
    display: flex;
    justify-content: flex-start;
    margin: 10px 0;
}

/* chat_input 커스터마이징 - 좌우 패딩 100px, 높이 100px */
.stChatInput {
    padding-left: 100px;
    padding-right: 100px;
}

</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# 3. 사이드바 내용 구현
with st.sidebar:
    # 왼쪽 사이드 바 상단에 '새 채팅'만 존재
    # 아이콘을 사용하려면 st.button에 emoji를 사용하거나, 더 커스텀된 컴포넌트가 필요
    if st.button("✨ 새 채팅", use_container_width=True):
        # 새 채팅 시작 시 현재 채팅 상태 초기화
        st.session_state.current_chat_id = None
        st.rerun()

    # 구분선
    st.markdown("---")

    # '내 채팅' 제목
    st.markdown("**내 채팅**")

    # 채팅 히스토리 표시 (첫 번째 질문만 표시)
    if st.session_state.chat_history:
        for idx, chat in enumerate(st.session_state.chat_history):
            first_question = chat.get("first_question", "")
            if first_question:
                # 채팅 항목을 버튼으로 표시
                if st.button(
                    first_question[:50] + ("..." if len(first_question) > 50 else ""),
                    key=f"chat_{idx}",
                    use_container_width=True,
                ):
                    st.session_state.current_chat_id = idx
                    st.rerun()


# 4. 중앙 콘텐츠 구현
# 현재 채팅이 없거나 초기 상태일 때만 초기 화면 표시
if st.session_state.current_chat_id is None:
    # 중앙에 "무엇을 도와드릴까요?" 문구를 표시 (첨부 파일의 중앙 문구)
    # CSS로 중앙 정렬했으므로, 이 문구가 중앙에 나타납니다.
    st.markdown(
        """
        <div style="text-align: center; font-size: 50px; color: #555;">
            무엇을 도와드릴까요?
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 5. 질문 입력 영역 (st.chat_input으로 Enter 키 제출 지원)
    question = st.chat_input(placeholder="무엇이든 물어보세요", key="question_input")

    # 질문 제출 처리 (Enter 키 자동 제출)
    if question and question.strip():
        # AI 응답 생성
        ai_response = generate_ai_response(question.strip())

        # 새로운 채팅 생성
        new_chat_id = len(st.session_state.chat_history)
        st.session_state.chat_history.append(
            {
                "first_question": question.strip(),
                "messages": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": ai_response},
                ],
            }
        )
        st.session_state.current_chat_id = new_chat_id
        st.rerun()

else:
    # 선택된 채팅이 있을 때 채팅 화면 표시
    chat = st.session_state.chat_history[st.session_state.current_chat_id]

    # 채팅 메시지 표시
    for msg in chat.get("messages", []):
        if msg["role"] == "user":
            # 사용자 메시지: 연한 보라색 배경의 둥근 상자, 오른쪽 정렬
            st.markdown(
                f"""<div class="user-message-container">
                <div class="user-message">{msg['content']}</div>
                </div>""",
                unsafe_allow_html=True,
            )
        else:
            # AI 응답: 왼쪽 정렬로 출력
            st.markdown(
                f"""<div class="ai-message-container">
                <div class="ai-message">{msg['content']}</div>
                </div>""",
                unsafe_allow_html=True,
            )

    # 질문 입력 영역 (st.chat_input으로 Enter 키 제출 지원)
    question = st.chat_input(
        placeholder="무엇이든 물어보세요", key="chat_question_input"
    )

    # 질문 제출 처리 (Enter 키 자동 제출)
    if question and question.strip():
        # AI 응답 생성
        ai_response = generate_ai_response(question.strip())

        # 현재 채팅에 메시지 추가
        st.session_state.chat_history[st.session_state.current_chat_id][
            "messages"
        ].append({"role": "user", "content": question})
        # AI 응답 추가
        st.session_state.chat_history[st.session_state.current_chat_id][
            "messages"
        ].append({"role": "assistant", "content": ai_response})
        st.rerun()
