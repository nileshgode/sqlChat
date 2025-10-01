# app.py

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from app.llm_config import get_llm
from app.utils import get_db_connection
from app.agent import create_sql_agent_graph

# --- Page Configuration ---
st.set_page_config(page_title="LangGraph SQL Agent", page_icon="🤖", layout="wide")
st.title("LangGraph SQL Agent 🤖")
st.write("Ask questions about your SQL database. The agent will show its work and provide a final answer.")

# --- Sidebar Configuration ---
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
        # 1. Initialize LLM and Database Connection
        llm = get_llm(llm_provider)
        db = get_db_connection(db_uri)

        if db:
            st.info(f"Connected to **{db.dialect}** database. Available tables: `{db.get_usable_table_names()}`")
            
            # 2. Create the LangGraph Agent
            agent_executor = create_sql_agent_graph(llm, db)

            # 3. Display Chat History
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # 4. Handle User Input and Agent Execution
            if question := st.chat_input("Ask a question about your database..."):
                # Add user question to history and display it
                st.session_state.messages.append({"role": "user", "content": question})
                with st.chat_message("user"):
                    st.markdown(question)

                # Execute the agent and stream the response
                with st.chat_message("assistant"):
                    # Use two columns for better layout: one for logs, one for the final answer
                    log_col, answer_col = st.columns(2)
                    
                    with log_col:
                        log_placeholder = st.empty()
                        log_output = "### Agent Execution Log\n\n---\n\n"
                    
                    final_step = None

                    # Stream the agent's execution to show the process
                    for step in agent_executor.stream({"messages": [("user", question)]}):
                        final_step = step # Always save the last step
                        
                        for key, value in step.items():
                            if key == "__end__":
                                continue
                            
                            # Append the current node's activity to the log
                            log_output += f"**Executing Node:** `{key}`\n\n"
                            messages = value.get("messages", [])
                            if messages:
                                message = messages[-1]
                                if message.content:
                                    log_output += f"**Content:** {message.content}\n\n"
                                if hasattr(message, 'tool_calls') and message.tool_calls:
                                    tool_info = message.tool_calls[0]
                                    log_output += f"**Tool Call:** `{tool_info['name']}` with args `{tool_info['args']}`\n\n"
                            log_output += "---\n\n"
                            log_placeholder.markdown(log_output)

                    # After the loop, extract and display the definitive final answer
                    with answer_col:
                        st.markdown("### Final Answer")
                        if final_step and "__end__" in final_step:
                            final_answer = final_step['__end__']['messages'][-1].content
                            st.markdown(final_answer)
                            # Add the final answer to the session state
                            st.session_state.messages.append({"role": "assistant", "content": final_answer})
                        else:
                            st.error("The agent finished without producing a final answer. Please check the terminal logs for details.")
        else:
            st.error("Failed to connect to the database. Please check the URI and try again.")
            
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.exception(e) # Also print the full traceback for detailed debugging

else:
    st.warning("Please provide a database URI in the sidebar to start.")
