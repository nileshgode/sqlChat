# app.py
import streamlit as st
from agent import MyAgent  # or whatever class
from llm_config import get_llm
from utils import connect_db  # or similar

provider = st.sidebar.selectbox("LLM Provider", ["ollama", "openai"])
model_name = st.sidebar.text_input("Model name", "llama3.1" if provider=="ollama" else "gpt-4")
temp = st.sidebar.slider("Temperature", 0.0, 1.0, 0.0)

db_uri = st.sidebar.text_input("Database URI", "sqlite:///Chinook.db")

llm = get_llm(provider, model_name=model_name, temperature=temp)
agent = MyAgent(llm=llm, db_uri=db_uri)

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.text_input("Ask a question:")

if user_input:
    # call agent pipeline
    answer, sql, result_df = agent.run(user_input)
    st.session_state.history.append((user_input, answer, sql, result_df))

for (q, ans, sql, df) in st.session_state.history:
    st.write("**You:**", q)
    st.write("**SQL:**", sql)
    st.write("**Answer:**", ans)
    st.dataframe(df)
