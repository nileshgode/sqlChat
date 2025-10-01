# app.py

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from llm_config import get_llm
from utils import get_db_connection
from agent import create_sql_agent_graph

# --- Page Configuration ---
st.set_page_config(page_title="LangGraph SQL Agent", page_icon="🤖")
st.title("LangGraph SQL Agent 🤖")
st.write("Ask questions about your SQL database. The agent will show its work.")

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("Configuration")
    llm_provider = st.selectbox("Choose LLM Provider", ["ollama", "openai"], index=0)
    db_uri = st.text_input(
        "Database URI",
        value="sqlite:///Chinook.db",
        help="Enter the connection string for your database."
    )

# --- Initialize Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Main Application Logic ---
if db_uri:
    try:
        llm = get_llm(llm_provider)
        db = get_db_connection(db_uri)

        if db:
            st.info(f"Connected to **{db.dialect}** DB. Tables: `{db.get_usable_table_names()}`")
            agent_executor = create_sql_agent_graph(llm, db)

            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            if question := st.chat_input("Ask your question..."):
                st.session_state.messages.append({"role": "user", "content": question})
                with st.chat_message("user"):
                    st.markdown(question)

                with st.chat_message("assistant"):
                    with st.spinner("The agent is thinking..."):
                        response_placeholder = st.empty()
                        intermediate_steps = ""
                        final_step = None

                        # Stream the agent's execution
                        for step in agent_executor.stream({"messages": [("user", question)]}):
                            # Save the last step to get the final answer after the loop
                            final_step = step
                            
                            # Display all intermediate steps
                            for key, value in step.items():
                                if key != "__end__":
                                    intermediate_steps += f"**Node:** `{key}`\n"
                                    messages = value.get("messages", [])
                                    if messages:
                                        message = messages[-1]
                                        if message.content:
                                            intermediate_steps += f"**Content:** {message.content}\n"
                                        if hasattr(message, 'tool_calls') and message.tool_calls:
                                            tool_info = message.tool_calls[0]
                                            intermediate_steps += f"**Tool Call:** `{tool_info['name']}` with args `{tool_info['args']}`\n"
                                    intermediate_steps += "---\n"
                                    response_placeholder.markdown(intermediate_steps)

                        # After the loop, extract the definitive final answer from the last step
                        if final_step and "__end__" in final_step:
                            final_answer = final_step['__end__']['messages'][-1].content
                            response_placeholder.markdown(final_answer)
                            st.session_state.messages.append({"role": "assistant", "content": final_answer})
                        else:
                            st.error("The agent finished without producing a final answer. Please check the agent's logic.")
        else:
            st.error("Failed to connect to the database. Please check the URI.")
            
    except Exception as e:
        st.error(f"An error occurred: {e}")

else:
    st.warning("Please provide a database URI in the sidebar.")
