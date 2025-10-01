# app.py

import streamlit as st
from app.llm_config import get_llm
from app.utils import get_db_connection
from app.agent import create_sql_agent_graph

# --- Page Configuration ---
st.set_page_config(page_title="Structured SQL Agent", page_icon="🤖", layout="wide")
st.title("Structured LangGraph SQL Agent 🤖")
st.write("Ask a question, and the agent will follow a strict workflow to get your answer.")

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("Configuration")
    llm_provider = st.selectbox("Choose LLM Provider", ["ollama", "openai"], index=0)
    db_uri = st.text_input("Database URI", value="sqlite:///Chinook.db")

# --- Initialize Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Main Application Logic ---
if db_uri:
    try:
        llm = get_llm(llm_provider)
        db = get_db_connection(db_uri)

        if db:
            st.info(f"Connected to **{db.dialect}** database.")
            agent_executor = create_sql_agent_graph(llm, db)

            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            if question := st.chat_input("Ask your question..."):
                st.session_state.messages.append({"role": "user", "content": question})
                with st.chat_message("user"):
                    st.markdown(question)

                with st.chat_message("assistant"):
                    with st.spinner("Agent is processing your request..."):
                        final_step = None
                        
                        # The input to the agent is now simpler
                        inputs = {"messages": [("user", question)]}
                        
                        # Stream the execution
                        for step in agent_executor.stream(inputs):
                            final_step = step
                            # Optional: log intermediate steps to the console if needed for debugging
                            # print(step)

                        # The final answer is in the last message of the final step
                        if final_step and "__end__" in final_step:
                            final_answer = final_step['__end__']['messages'][-1].content
                            st.markdown(final_answer)
                            st.session_state.messages.append({"role": "assistant", "content": final_answer})
                        else:
                            st.error("The agent finished unexpectedly. Check terminal logs.")
        else:
            st.error("Failed to connect to the database.")
            
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.exception(e)
else:
    st.warning("Please provide a database URI to start.")

