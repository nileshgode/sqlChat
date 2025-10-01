# app.py

import streamlit as st
from app.llm_config import get_llm
from app.utils import get_db_connection
from app.agent import create_sql_agent_graph

st.set_page_config(page_title="LangGraph SQL Agent", page_icon="🤖")
st.title("LangGraph SQL Agent 🤖")
st.write("Ask questions about your SQL database in plain English. The agent will figure out the rest!")

# --- Sidebar for Configuration ---
with st.sidebar:
    st.header("Configuration")
    llm_provider = st.selectbox("Choose LLM Provider", ["ollama", "openai"])
    db_uri = st.text_input("Database URI", value="sqlite:///Chinook.db")

# --- Main Application ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if db_uri:
    try:
        llm = get_llm(llm_provider)
        db = get_db_connection(db_uri)

        if db:
            st.info(f"Connected to **{db.dialect}** database. Tables: `{db.get_usable_table_names()}`")
            agent_executor = create_sql_agent_graph(llm, db)

            question = st.chat_input("Ask a question about the database...")
            if question:
                st.session_state.messages.append({"role": "user", "content": question})
                with st.chat_message("user"):
                    st.markdown(question)

                with st.chat_message("assistant"):
                    with st.spinner("Agent is thinking..."):
                        response_placeholder = st.empty()
                        final_response = ""
                        
                        # Stream agent's thought process
                        for step in agent_executor.stream({"messages": [("user", question)]}):
                            if "__end__" not in step:
                                for key, value in step.items():
                                    role = value['messages'][-1].role
                                    content = value['messages'][-1].content
                                    if content:
                                        final_response += f"**{role.capitalize()} says:** {content}\n\n"
                                    if hasattr(value['messages'][-1], 'tool_calls') and value['messages'][-1].tool_calls:
                                        final_response += f"**Tool Call:** `{value['messages'][-1].tool_calls[0]['name']}` with args `{value['messages'][-1].tool_calls[0]['args']}`\n\n"
                                    response_placeholder.markdown(final_response)
                        
                        # Extract final answer from the last message
                        final_answer = step['__end__']['messages'][-1].content
                        response_placeholder.markdown(final_answer)
                        st.session_state.messages.append({"role": "assistant", "content": final_answer})

        else:
            st.error("Failed to connect to the database. Please check the URI.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.warning("Please provide a database URI in the sidebar to start.")

