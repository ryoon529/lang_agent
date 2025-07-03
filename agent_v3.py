import ast
from langchain_community.tools import TavilySearchResults  
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.tools.retriever import create_retriever_tool
from dotenv import load_dotenv
import streamlit as st
from pinecone import Pinecone
import os

load_dotenv()

TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

# Pinecone 벡터 스토어 설정
embedding = OpenAIEmbeddings(model='text-embedding-3-large')
index_name = "school-carbon-neutral"

# Pinecone 초기화 및 벡터 스토어 연결
pc = Pinecone(api_key=PINECONE_API_KEY)
vectorstore = PineconeVectorStore(
    index_name=index_name,
    embedding=embedding
)

# 이미 만들어져 있는 vectorstore로 retriever 만들기
retriever = vectorstore.as_retriever(k=3)

#retriever를 agent가 사용할 수 있도록 툴로 변환시키기
retriever_tool = create_retriever_tool(
    retriever,
    "search_pdf",
    "학교에서의 탄소 중립과 관련된 정보를 검색하고 반환합니다. 학교 탄소 중립 관련 질문에 사용하세요."
)

@tool
def calculate(query: str) -> str:
    """계산기. 수식만 입력받습니다."""
    return ast.literal_eval(query)

#웹 서치를 위한 타빌리 함수 사용
search = TavilySearchResults(api_key=TAVILY_API_KEY)

# 사용가능한 툴을 tools 리스트에 추가. 총 3개의 툴 사용
tools = [calculate, search, retriever_tool]
# 챗 모델 생성
model = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
# create_react_agent 함수로 그래프를 생성 
agent_executor = create_react_agent(model, tools)

# 스트림릿 UI
st.title("토의, 토론 에이전트 (Pinecone)")
st.write("주제에 맞는 의견을 나누어 보자.")

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = [SystemMessage(
        content="""
        당신은 학생들과 함께 토의, 토론에서 의견을 낼 수 있는 또래 학습자입니다. 
        학생이 자신의 의견을 만들 수 있도록 단계적으로 안내합니다. 
        1. 학생들의 질문에 대해 같은 또래 학습자의 입장에서 대답합니다.
        2. 초등학생 5~6학년의 수준에서 답변합니다.
        3. 필요한 경우 retriever_tool을 사용하여 답변합니다.
        4. 인터넷 검색이 필요한 경우 search_tool을 사용합니다.
        5. 계산이 필요한 경우 calculate_tool을 사용합니다.
        """
    )]

# 대화 기록 표시
for message in st.session_state.messages[1:]:  # 시스템 메시지 제외
    if hasattr(message, 'content'):
        if 'HumanMessage' in str(type(message)):
            st.chat_message("user").write(message.content)
        elif 'AIMessage' in str(type(message)):
            st.chat_message("assistant").write(message.content)

# 사용자 입력
if prompt := st.chat_input("의견을 나누어 보자."):
    # 사용자 메시지 추가
    st.session_state.messages.append(HumanMessage(content=prompt))
    st.chat_message("user").write(prompt)
    
    # AI 응답 생성
    with st.chat_message("assistant"):
        with st.spinner("내 생각을 말해줄게..."):
            try:
                response = agent_executor.invoke({"messages": st.session_state.messages})
                ai_message = response['messages'][-1]
                st.write(ai_message.content)
                st.session_state.messages.append(ai_message)
            except Exception as e:
                st.error(f"오류가 발생했습니다: {e}") 