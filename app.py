import streamlit as st
import asyncio
import pandas as pd
import sqlite3
import datetime
from langchain_community.utilities import SQLDatabase
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMessageTermination, MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import SelectorGroupChat
from autogen_core.memory import ListMemory, MemoryContent, MemoryMimeType
from autogen_ext.models.openai import OpenAIChatCompletionClient

# 페이지 설정
st.set_page_config(page_title="건축사고 분석 시스템", layout="wide")

# 세션 상태 초기화
if "initialized" not in st.session_state:
    st.session_state.initialized = False
    st.session_state.api_key = ""
    st.session_state.team = None
    st.session_state.messages = []
    st.session_state.loading = False

# 현재시간 출력 함수
def get_current_time() -> str:
    """Return the current time as a string."""
    return str(datetime.datetime.now())

# DB request하는 함수
def db_request(query: str):
    """
    Execute SQL query and return results.
    """
    try:
        conn = sqlite3.connect('construction_accidents.db')
        result = pd.read_sql_query(query, conn)
        conn.close()
        return result
    except Exception as e:
        return f"Query execution failed: {e}"

# 건축사고 분류표 데이터
def get_accident_classification():
    return [
        {"공종": "건축 > 가설공사", "사고객체": "건설공구 > 사다리", "작업프로세스": "기타"},
        {"공종": "건축 > 가설공사", "사고객체": "건설공구 > 사다리", "작업프로세스": "마감작업"},
        {"공종": "건축 > 가설공사", "사고객체": "건설공구 > 사다리", "작업프로세스": "설비작업"},
        {"공종": "건축 > 가설공사", "사고객체": "건설공구 > 사다리", "작업프로세스": "설치작업"},
        {"공종": "건축 > 가설공사", "사고객체": "건설공구 > 사다리", "작업프로세스": "연결작업"},
        {"공종": "건축 > 가설공사", "사고객체": "건설공구 > 사다리", "작업프로세스": "이동"},
        {"공종": "건축 > 가설공사", "사고객체": "건설공구 > 사다리", "작업프로세스": "조립작업"},
        {"공종": "건축 > 가설공사", "사고객체": "건설공구 > 사다리", "작업프로세스": "준비작업"},
        {"공종": "건축 > 가설공사", "사고객체": "건설공구 > 사다리", "작업프로세스": "해체작업"},
        {"공종": "건축 > 가설공사", "사고객체": "건설공구 > 사다리", "작업프로세스": "확인 및 점검작업"}
    ]

# 분류표 메모리 생성 함수
async def create_classification_memory():
    construction_accident_memory = ListMemory()
    
    # 표 형태로 구조화
    classification_text = "# 건축사고 분류표\n\n"
    classification_text += "| 공종 | 사고객체 | 작업프로세스 |\n"
    classification_text += "|------|----------|-------------|\n"
    
    for item in get_accident_classification():
        classification_text += f"| {item['공종']} | {item['사고객체']} | {item['작업프로세스']} |\n"
    
    await construction_accident_memory.add(
        MemoryContent(
            content=classification_text,
            mime_type=MemoryMimeType.TEXT
        )
    )
    
    return construction_accident_memory

# 테이블 메타데이터 메모리 생성 함수
async def create_table_memory():
    accidentcase_table_memory = ListMemory()
    
    try:
        db = SQLDatabase.from_uri("sqlite:///construction_accidents.db")
        table_info = str(db.get_table_info())
    except Exception as e:
        table_info = "Error loading table metadata: " + str(e)
    
    await accidentcase_table_memory.add(
        MemoryContent(
            content=table_info,
            mime_type=MemoryMimeType.TEXT
        )
    )
    
    return accidentcase_table_memory

# 에이전트 팀 초기화 함수
async def initialize_team(api_key):
    # 모델 클라이언트 설정
    model_client = OpenAIChatCompletionClient(
        model="gpt-4o",
        api_key=api_key
    )
    
    model_client_reason = OpenAIChatCompletionClient(
        model="o3-mini",
        api_key=api_key
    )
    
    # 메모리 초기화
    construction_accident_memory = await create_classification_memory()
    accidentcase_table_memory = await create_table_memory()
    
    # 에이전트 생성
    sql_assistant = AssistantAgent(
        "sql_assistant",
        model_client=model_client,
        memory=[accidentcase_table_memory],
        tools=[db_request],
        system_message="""당신은 SQL을 사용자의 요청에 맞게 작성하는 어시스턴트입니다. 
당신은 도구를 사용하여 SQL쿼리를 DB로 전송할 수 있습니다.
결과를 명확하게 표시하고 DB 오류가 발생하면 쿼리를 수정하세요.
- 공사종류는 '사고분류.공종'컬럼에서 찾아주세요.
- 공사장비는 '사고분류.사고객체'컬럼에서 찾아주세요.
- 작업내용는 '사고분류.작업프로세스'컬럼에서 찾아주세요.
"""
    )
    
    construction_expert = AssistantAgent(
        name="construction_expert",
        model_client=model_client_reason,
        memory=[construction_accident_memory],
        system_message="""당신은 건축사고 분류 전문가입니다.
제공된 건축사고 분류표를 기반으로 사용자의 질문을 분류해 주세요.
특정 작업이나 도구에 대한 질문을 받으면, 해당 분류항목을 찾아 알려주세요.
답변할 수 없는 내용에 대해서는 정직하게 모른다고 답변하세요.
결과는 공종, 사고객체, 작업프로세스 형식으로 명확히 구분하여 제시하세요.
"""
    )
    
    planning_agent = AssistantAgent(
        "PlanningAgent",
        description="An agent for planning tasks, this agent should be the first to engage when given a new task.",
        model_client=model_client,
        system_message="""당신은 계획 수립 에이전트입니다.
당신의 역할은 복잡한 작업을 더 작고 관리 가능한 하위 작업으로 나누고, 진행 상황을 모니터링하는 것입니다.

당신의 팀 구성원은 다음과 같습니다:
    construction_expert: 건축사고 분류 전문가로, 사용자의 질문을 분류표에 따라 분석하고 공종/사고객체/작업프로세스를 식별합니다.
    sql_assistant: DB 쿼리 전문가로, construction_expert가 식별한 분류 기준에 따라 DB에서 사고 사례를 검색합니다.

작업 진행 순서:
1. 새 작업을 받으면, 먼저 sql_assistant에게 DB 테이블 메타데이터 조회를 요청하세요.
2. construction_expert에게 사용자 질문을 분석하고 관련 분류(공종/사고객체/작업프로세스)를 식별하도록 요청하세요.
3. construction_expert의 분류 결과를 바탕으로, sql_assistant에게 해당 분류에 맞는 사고 사례를 조회하는 SQL 쿼리를 작성하고 실행하도록 요청하세요.
4. 결과를 검토하고 필요시 sql_assistant에게 쿼리 수정을 요청하세요.
5. 최종 결과를 사용자에게 명확하게 요약해서 보고하세요.

작업을 할당할 때는 다음 형식을 사용하세요:
1. <agent> : <task>

모든 작업이 성공적으로 완료되었고 사용자의 질문에 충분히 답변했을 때만 'TERMINATE'를 메시지에 포함하세요.
그 외에는 절대 'TERMINATE'를 사용하지 마세요."""
    )
    
    # 종료 조건 설정
    text_mention_termination = TextMentionTermination("TERMINATE")
    max_messages_termination = MaxMessageTermination(max_messages=25)
    termination = text_mention_termination | max_messages_termination
    
    # 셀렉터 프롬프트 설정
    selector_prompt = """대화 내용을 분석하고 다음 작업을 수행할 적절한 에이전트를 선택하세요.

에이전트 역할:
{roles}

현재 대화 맥락:
{history}

작업 흐름 규칙:
1. 새로운 작업이나 질문이 있을 때는 항상 PlanningAgent가 먼저 작업을 분석하고 할당해야 합니다.
2. 작업 할당 후에는 지정된 에이전트만 작업을 수행해야 합니다.
3. PlanningAgent가 작업을 할당한 후에는 해당 에이전트를 선택하세요.
4. 에이전트가 작업을 완료한 후에는 PlanningAgent를 선택하여 결과를 검토하고 다음 단계를 결정하게 하세요.
5. 작업 흐름: PlanningAgent(계획) → 지정된 에이전트(실행) → PlanningAgent(검토) → 다음 에이전트(실행) → ...

위 대화 내용을 읽고, {participants} 중에서 다음 작업을 수행할 가장 적합한 에이전트 하나를 선택하세요.

다음과 같은 경우에만 TERMINATE를 출력하세요:
- PlanningAgent가 최종 결과를 요약하고 'TERMINATE'를 포함했을 때
- 더 이상 할당하거나 수행할 작업이 없고 사용자의 질문이 완전히 해결되었을 때"""
    
    # 팀 생성
    team = SelectorGroupChat(
        [construction_expert, sql_assistant, planning_agent],
        model_client=model_client,
        termination_condition=termination,
        selector_prompt=selector_prompt,
        allow_repeated_speaker=True,
    )
    
    return team

# 팀 초기화 함수
async def reset_team():
    if st.session_state.team:
        await st.session_state.team.reset()
    st.session_state.team = await initialize_team(st.session_state.api_key)

# 챗봇 응답 출력을 위한 함수
async def process_response(task):
    st.session_state.loading = True
    
    try:
        # 사용자 메시지 추가
        st.session_state.messages.append({"role": "user", "content": task})
        
        # 팀이 없는 경우 초기화
        if st.session_state.team is None:
            st.session_state.team = await initialize_team(st.session_state.api_key)
        
        # 팀 응답 가져오기 - 스트리밍 모드 사용
        # 처리된 메시지를 추적하기 위한 집합
        processed_messages = set()
        
        async for result in st.session_state.team.run_stream(task=task):
            # 최신 메시지 확인
            if hasattr(result, 'messages'):
                # 새 메시지만 처리
                for msg in result.messages:
                    # 메시지 고유 ID 생성 (여기서는 source와 content 조합)
                    if hasattr(msg, 'content') and msg.content:
                        msg_id = f"{msg.source}_{msg.content[:50]}"
                        
                        # # 이미 처리한 메시지는 건너뛰기
                        # if msg_id in processed_messages:
                        #     continue
                        
                        # 사용자 메시지는 건너뛰기
                        if msg.source == "user":
                            processed_messages.add(msg_id)
                            continue
                        
                        # # 도구 호출 메시지 처리
                        # if hasattr(msg, 'type') and msg.type in ('ToolCallRequestEvent', 'ToolCallExecutionEvent', 'ToolCallSummaryMessage'):
                        #     processed_messages.add(msg_id)
                        #     continue
                        
                        # 에이전트 이름과 응답 내용 표시
                        with st.chat_message("assistant"):
                            st.markdown(f"**{msg.source}**: {msg.content}")
                        
                        # 메시지 히스토리에 추가
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": f"**{msg.source}**: {msg.content}"
                        })
                        
                        # 처리된 메시지 표시
                        processed_messages.add(msg_id)
        
    except Exception as e:
        st.session_state.messages.append({"role": "assistant", "content": f"오류가 발생했습니다: {str(e)}"})
        with st.chat_message("assistant"):
            st.error(f"오류가 발생했습니다: {str(e)}")
    
    st.session_state.loading = False

# 메인 UI
st.title("건축사고 분석 시스템")

# API 키 입력
if not st.session_state.initialized:
    with st.form("api_key_form"):
        api_key = st.text_input("OpenAI API 키를 입력하세요:", type="password")
        submit_button = st.form_submit_button("시작하기")
        
        if submit_button and api_key:
            st.session_state.api_key = api_key
            st.session_state.initialized = True
            st.rerun()

# 메인 인터페이스
if st.session_state.initialized:
    # 사이드바에 데이터베이스 상태 표시
    with st.sidebar:
        st.header("시스템 정보")
        try:
            db = SQLDatabase.from_uri("sqlite:///construction_accidents.db")
            tables = db.get_table_info().split('\n')
            st.subheader("데이터베이스 테이블")
            for table in tables:
                if table.strip():
                    st.text(table)
            
            # DB 상태 확인 및 표시
            try:
                conn = sqlite3.connect('construction_accidents.db')
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                table_list = cursor.fetchall()
                conn.close()
                
                if table_list:
                    st.success("데이터베이스 연결 성공")
                else:
                    st.warning("데이터베이스에 테이블이 없습니다")
            except Exception as e:
                st.error(f"데이터베이스 연결 오류: {str(e)}")
        except Exception as e:
            st.error(f"데이터베이스 정보 로드 오류: {str(e)}")
        
        st.subheader("분류표")
        with st.expander("건축사고 분류표 보기"):
            df = pd.DataFrame(get_accident_classification())
            st.table(df)
        
        if st.button("팀 초기화"):
            asyncio.run(reset_team())
            st.success("팀이 초기화되었습니다!")

    # 채팅 인터페이스
    st.header("건축사고 관련 질문하기")
    
    # 채팅 메시지 표시 - 이전 메시지만 표시하고 새 메시지는 스트리밍으로 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # 로딩 표시
    if st.session_state.loading:
        with st.chat_message("assistant"):
            st.write("에이전트 팀이 작업을 처리하고 있습니다...")
    
    # 사용자 입력
    if prompt := st.chat_input("질문을 입력하세요..."):
        # 사용자 메시지 표시
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # 응답 비동기 처리
        asyncio.run(process_response(prompt))

# 실행 안내
if not st.session_state.initialized:
    st.info("시작하려면 OpenAI API 키를 입력하세요.")
    
    with st.expander("사용 안내"):
        st.markdown("""
        ### 건축사고 분석 시스템 사용 방법
        
        1. OpenAI API 키를 입력하여 시스템을 시작합니다.
        2. 건축사고 관련 질문을 입력하면 다음과 같은 과정으로 처리됩니다:
           - 계획 에이전트가 작업을 분석하고 할당합니다.
           - 건축사고 분류 전문가가 질문을 분류합니다.
           - SQL 어시스턴트가 데이터베이스에서 관련 사고 사례를 검색합니다.
           - 계획 에이전트가 결과를 종합하여 답변을 제공합니다.
        3. 사이드바에서 데이터베이스 정보와 분류표를 확인할 수 있습니다.
        4. 문제가 발생하면 '팀 초기화' 버튼을 클릭하세요.
        
        ### 질문 예시
        - "사다리를 이용한 작업 중 발생한 사고 사례가 있나요?"
        - "지게차 관련 사고 사례를 알려주세요."
        - "건축 가설공사 중 발생한 사고 통계를 알려주세요."
        """)