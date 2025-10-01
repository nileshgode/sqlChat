# app.py

# app.py

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from llm_config import get_llm
from utils import get_db_connection
from agent import create_sql_agent_graph

# --- Page Configuration ---
st.set_page_config(page_title="LangGraph SQL Agent", page_icon="🤖")
st.title("LangGraph SQL Agent 🤖")
st.write("Ask questions about your SQL database in plain English. The agent will figure out the rest!")

# --- Sidebar for Configuration ---
with st.sidebar:
    st.header("Configuration")
    llm_provider = st.selectbox("Choose LLM Provider", ["ollama", "openai"], index=0)
    db_uri = st.text_input(
        "Database URI",
        value="sqlite:///Chinook.db",
        help="Enter the connection string for your database (e.g., 'sqlite:///Chinook.db')."
    )

# --- Initialize Session State for Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Main Application Logic ---
if db_uri:
    try:
        llm = get_llm(llm_provider)
        db = get_db_connection(db_uri)

        if db:
            st.info(f"Connected to **{db.dialect}** database. Available tables: `{db.get_usable_table_names()}`")
            agent_executor = create_sql_agent_graph(llm, db)

            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            if question := st.chat_input("Ask a question about the database..."):
                st.session_state.messages.append({"role": "user", "content": question})
                with st.chat_message("user"):
                    st.markdown(question)

                with st.chat_message("assistant"):
                    with st.spinner("The agent is thinking..."):
                        response_placeholder = st.empty()
                        final_response_stream = ""
                        final_step = None

                        for step in agent_executor.stream({"messages": [("user", question)]}):
                            # Store the last step to access the final answer later
                            final_step = step
                            if "__end__" in step:
                                continue # We'll process the end step after the loop

                            for key, value in step.items():
                                messages = value.get("messages", [])
                                if not messages:
                                    continue

                                message = messages[-1]
                                
                                # Determine role safely
                                if isinstance(message, AIMessage): role_name = "Assistant"
                                elif isinstance(message, HumanMessage): role_name = "User"
                                elif isinstance(message, ToolMessage): role_name = "Tool Output"
                                else: role_name = "System"
                                
                                # Safely build the display string
                                if message.content:
                                    final_response_stream += f"**{role_name}:**\n{message.content}\n\n"
                                
                                if hasattr(message, 'tool_calls') and message.tool_calls:
                                    tool_info = message.tool_calls[0]
                                    final_response_stream += f"**Tool Call:** `{tool_info['name']}` with args `{tool_info['args']}`\n\n"
                                
                                response_placeholder.markdown(final_response_stream)
                        
                        # Now, safely access the final answer from the stored last step
                        if final_step and "__end__" in final_step:
                            final_answer = final_step['__end__']['messages'][-1].content
                            response_placeholder.markdown(final_answer)
                            st.session_state.messages.append({"role": "assistant", "content": final_answer})
                        else:
                            st.error("The agent finished without providing a final answer.")

        else:
            st.error("Failed to connect to the database. Please check the URI and try again.")
            
    except Exception as e:
        st.error(f"An error occurred: {e}")

else:
    st.warning("Please provide a database URI in the sidebar to start.")

