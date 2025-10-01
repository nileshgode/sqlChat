# app/app.py

import streamlit as st
from langchain_core.messages import HumanMessage
from utils import get_db_connection
from agent import create_sql_agent_graph
from llm_config import get_llm

st.set_page_config(page_title="SQL Chatbot", layout="wide")
st.title("💬 SQL Chat with LLM + LangGraph")

# --- Sidebar Config ---
st.sidebar.header("⚙️ Settings")
provider = st.sidebar.selectbox("LLM Provider", ["ollama", "openai"])
model_name = st.sidebar.text_input(
    "Model name",
    "llama3.1" if provider == "ollama" else "gpt-4"
)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.0)
db_uri = st.sidebar.text_input("Database URI", "sqlite:///Chinook.db")

# --- Initialize ---
if "agent" not in st.session_state:
    db = get_db_connection(db_uri)
    if db is None:
        st.error("❌ Could not connect to database.")
        st.stop()

    llm = get_llm(provider, model_name=model_name, temperature=temperature)
    agent = create_sql_agent_graph(llm, db)
    st.session_state.agent = agent
    st.session_state.history = []

# --- Chat Input ---
user_input = st.text_input("Ask me anything about the database:")

if user_input:
    with st.spinner("🔍 Thinking..."):
        initial_state = {
            "user_question": user_input,
            "messages": [HumanMessage(content=user_input)]
        }

        result_state = st.session_state.agent.invoke(initial_state)

        sql_query = result_state.get("sql_query", "")
        query_result = result_state.get("query_result", "")
        final_answer = result_state.get("final_answer", "")

        st.session_state.history.append({
            "question": user_input,
            "sql": sql_query,
            "result": query_result,
            "answer": final_answer
        })

# --- Display Chat History ---
for chat in st.session_state.history[::-1]:
    with st.expander(f"❓ {chat['question']}"):
        if chat["sql"]:
            st.markdown(f"**Generated SQL:**\n```sql\n{chat['sql']}\n```")
        if chat["result"]:
            st.markdown(f"**Query Result:**\n{chat['result']}")
        st.markdown(f"**Final Answer:**\n{chat['answer']}")
