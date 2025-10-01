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
        llm = get_llm(llm_provider)
        db = get_db_connection(db_uri)

        if db:
            st.info(f"Connected to **{db.dialect}** database. Available tables: `{db.get_usable_table_names()}`")
            agent_executor = create_sql_agent_graph(llm, db)

            # Display chat history
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Handle user input
            if question := st.chat_input("Ask a question about the database..."):
                st.session_state.messages.append({"role": "user", "content": question})
                with st.chat_message("user"):
                    st.markdown(question)

                # Execute agent and stream response
                with st.chat_message("assistant"):
                    with st.spinner("The agent is thinking..."):
                        response_placeholder = st.empty()
                        intermediate_stream = ""
                        final_answer = ""

                        # Stream the agent's execution steps
                        for step in agent_executor.stream({"messages": [("user", question)]}):
                            for key, value in step.items():
                                if key == "__end__":
                                    continue # Skip the end key, we look for the specific node

                                # --- This is the crucial change ---
                                # Check if the current step is our designated final response node
                                if key == "generate_final_response":
                                    messages = value.get("messages", [])
                                    if messages:
                                        final_answer = messages[-1].content
                                    continue # Stop processing this step further

                                # --- Logic to display intermediate steps ---
                                messages = value.get("messages", [])
                                if not messages:
                                    continue

                                message = messages[-1]
                                
                                # Determine role for display
                                if isinstance(message, AIMessage): role_name = "Assistant"
                                elif isinstance(message, HumanMessage): role_name = "User"
                                elif isinstance(message, ToolMessage): role_name = "Tool Output"
                                else: role_name = "System"
                                
                                # Build the intermediate stream for display
                                if message.content:
                                    intermediate_stream += f"**{role_name}:**\n{message.content}\n\n"
                                
                                if hasattr(message, 'tool_calls') and message.tool_calls:
                                    tool_info = message.tool_calls[0]
                                    intermediate_stream += f"**Tool Call:** `{tool_info['name']}` with args `{tool_info['args']}`\n\n"
                                
                                response_placeholder.markdown(intermediate_stream)
                        
                        # After the loop, display the captured final answer
                        if final_answer:
                            response_placeholder.markdown(final_answer)
                            st.session_state.messages.append({"role": "assistant", "content": final_answer})
                        else:
                            # This error should now be much rarer
                            st.error("The agent finished but failed to generate a final answer.")
        else:
            st.error("Failed to connect to the database. Please check the URI and try again.")
            
    except Exception as e:
        st.error(f"An error occurred: {e}")

else:
    st.warning("Please provide a database URI in the sidebar to start.")
