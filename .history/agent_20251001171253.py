# app/agent.py

import logging
from typing import Literal
from langchain_core.messages import AIMessage, BaseMessage
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langgraph.graph import MessagesState

# --- Add this logging configuration at the top ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

llm = None

def generate_final_response(state: MessagesState):
    # ... (no changes needed in this function from the previous version)
    logger.info("Entering generate_final_response node")
    system_prompt = (
        "You are a helpful assistant. Based on the entire conversation history, "
        "provide a clear and concise final answer to the user's original question. "
        "Synthesize the information from any tool outputs into a natural language response."
    )
    final_prompt_messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    response = llm.invoke(final_prompt_messages)
    logger.info(f"Final response generated: {response.content}")
    return {"messages": [response]}


def create_sql_agent_graph(llm_instance, db):
    global llm
    llm = llm_instance

    # ... (rest of the setup is the same)
    
    # --- THIS IS THE CRITICAL CHANGE ---
    # We are removing the conditional logic and making the flow explicit.
    # The agent will now always follow a set path.

    # Build the Graph
    builder = StateGraph(MessagesState)
    builder.add_node("list_tables", list_tables)
    builder.add_node("call_get_schema", call_get_schema)
    builder.add_node("get_schema", ToolNode([get_schema_tool]))
    builder.add_node("generate_query", generate_query) # This node now just generates the query
    builder.add_node("check_query", check_query)
    builder.add_node("run_query", ToolNode([run_query_tool]))
    builder.add_node("generate_final_response", generate_final_response)

    # --- EXPLICIT GRAPH FLOW ---
    builder.add_edge(START, "list_tables")
    builder.add_edge("list_tables", "call_get_schema")
    builder.add_edge("call_get_schema", "get_schema")
    builder.add_edge("get_schema", "generate_query")
    
    # The flow is no longer conditional. It always follows this path.
    builder.add_edge("generate_query", "check_query")
    builder.add_edge("check_query", "run_query")
    builder.add_edge("run_query", "generate_final_response")
    builder.add_edge("generate_final_response", END)
    
    return builder.compile()

# ... (the rest of your node functions: list_tables, call_get_schema, etc. remain the same)

def list_tables(state: MessagesState):
    logger.info("Entering list_tables node")
    #... function body
    return {"messages": [tool_call_message, tool_message, response]}

def call_get_schema(state: MessagesState):
    logger.info("Entering call_get_schema node")
    #... function body
    return {"messages": [response]}

def generate_query(state: MessagesState):
    logger.info("Entering generate_query node")
    #... function body
    return {"messages": [response]}

def check_query(state: MessagesState):
    logger.info("Entering check_query node")
    #... function body
    return {"messages": [response]}


