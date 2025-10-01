# app.py

# app.py

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from app.llm_config import get_llm
from app.utils import get_db_connection
from app.agent import create_sql_agent_graph

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
            if question := st.chat_input("Ask a question about the database..."):
                # Add user question to chat history and display it
                st.session_state.messages.append({"role": "user", "content": question})
                with st.chat_message("user"):
                    st.markdown(question)

                # Execute the agent and stream the response
                with st.chat_message("assistant"):
                    with st.spinner("The agent is thinking..."):
                        response_placeholder = st.empty()
                        final_response_stream = ""
                        
                        # Stream the agent's thought process and actions
                        for step in agent_executor.stream({"messages": [("user", question)]}):
                            if "__end__" not in step:
                                for key, value in step.items():
                                    message = value['messages'][-1]
                                    
                                    # Determine the role for display based on the message type
                                    if isinstance(message, AIMessage):
                                        role_name = "Assistant"
                                    elif isinstance(message, HumanMessage):
                                        role_name = "User"
                                    elif isinstance(message, ToolMessage):
                                        role_name = "Tool Output"
                                    else:
                                        role_name = "System"

                                    # Build the display string for the stream
                                    if message.content:
                                        final_response_stream += f"**{role_name}:**\n{message.content}\n\n"
                                    
                                    # Display tool call information
                                    if hasattr(message, 'tool_calls') and message.tool_calls:
                                        tool_info = message.tool_calls[0]
                                        final_response_stream += f"**Tool Call:** `{tool_info['name']}` with args `{tool_info['args']}`\n\n"
                                    
                                    response_placeholder.markdown(final_response_stream)
                        
                        # Extract and display the final answer
                        final_answer = step['__end__']['messages'][-1].content
                        response_placeholder.markdown(final_answer)
                        
                        # Add the final answer to the chat history
                        st.session_state.messages.append({"role": "assistant", "content": final_answer})

        else:
            st.error("Failed to connect to the database. Please check the URI and try again.")
            
    except Exception as e:
        st.error(f"An error occurred: {e}")

else:
    st.warning("Please provide a database URI in the sidebar to start.")

