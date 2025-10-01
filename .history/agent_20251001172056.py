# app/agent.py

import logging
from typing import Literal
from langchain_core.messages import AIMessage, BaseMessage
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langgraph.graph import MessagesState

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Module-level variables ---
llm = None
tools = []
get_schema_tool = None
run_query_tool = None
list_tables_tool = None

# --- Node Functions ---

def generate_final_response(state: MessagesState):
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

def list_tables(state: MessagesState):
    logger.info("Entering list_tables node")
    tool_call = {"name": "sql_db_list_tables", "args": {}, "id": "abc123", "type": "tool_call"}
    tool_call_message = AIMessage(content="", tool_calls=[tool_call])
    tool_message = list_tables_tool.invoke(tool_call)
    response = AIMessage(f"Available tables: {tool_message.content}")
    return {"messages": [tool_call_message, tool_message, response]}

def call_get_schema(state: MessagesState):
    logger.info("Entering call_get_schema node")
    llm_with_tools = llm.bind_tools([get_schema_tool], tool_choice="any")
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def generate_query(state: MessagesState):
    logger.info("Entering generate_query node")
    generate_query_system_prompt = "You are an agent designed to interact with a SQL database. Given an input question, create a syntactically correct query to run."
    system_message = {"role": "system", "content": generate_query_system_prompt}
    llm_with_tools = llm.bind_tools([run_query_tool])
    response = llm_with_tools.invoke([system_message] + state["messages"])
    logger.info(f"LLM output from generate_query: {response}")
    return {"messages": [response]}

# --- THIS IS THE CRITICAL FIX ---
def check_query(state: MessagesState):
    logger.info("Entering check_query node")
    
    last_message = state["messages"][-1]
    query_text = ""

    # Resiliently find the SQL query from the last message
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        # Ideal case: Model used tool_calls correctly
        logger.info("Query found in tool_calls.")
        query_text = last_message.tool_calls[0]['args']['query']
    elif isinstance(last_message, AIMessage) and last_message.content:
        # Fallback case: Model put the SQL query in the content field
        logger.warning("tool_calls not found. Falling back to message content for the query.")
        query_text = last_message.content
    
    if not query_text:
        # If we still can't find a query, we cannot proceed.
        error_message = AIMessage(content="Error: Could not find a SQL query to execute in the previous step.")
        logger.error("Failed to extract SQL query from the last AI message.")
        return {"messages": [error_message]}

    # Now, proceed with the check using the extracted query_text
    check_query_system_prompt = "You are a SQL expert. Double check the given query for common mistakes. If there are no mistakes, just reproduce the original query. Your output must be a tool call to run the query."
    system_message = {"role": "system", "content": check_query_system_prompt}
    user_message = {"role": "user", "content": query_text}
    
    llm_with_tools = llm.bind_tools([run_query_tool], tool_choice="any")
    response = llm_with_tools.invoke([system_message, user_message])
    
    # Preserve the ID if it exists, for tracing purposes
    if hasattr(last_message, 'id') and hasattr(response, 'id'):
        response.id = last_message.id
        
    return {"messages": [response]}

# --- Graph Builder Function ---
def create_sql_agent_graph(llm_instance, db):
    global llm, tools, get_schema_tool, run_query_tool, list_tables_tool
    llm = llm_instance

    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()
    get_schema_tool = next(tool for tool in tools if tool.name == "sql_db_schema")
    run_query_tool = next(tool for tool in tools if tool.name == "sql_db_query")
    list_tables_tool = next(tool for tool in tools if tool.name == "sql_db_list_tables")

    builder = StateGraph(MessagesState)
    builder.add_node("list_tables", list_tables)
    builder.add_node("call_get_schema", call_get_schema)
    builder.add_node("get_schema", ToolNode([get_schema_tool]))
    builder.add_node("generate_query", generate_query)
    builder.add_node("check_query", check_query)
    builder.add_node("run_query", ToolNode([run_query_tool]))
    builder.add_node("generate_final_response", generate_final_response)

    builder.add_edge(START, "list_tables")
    builder.add_edge("list_tables", "call_get_schema")
    builder.add_edge("call_get_schema", "get_schema")
    builder.add_edge("get_schema", "generate_query")
    builder.add_edge("generate_query", "check_query")
    builder.add_edge("check_query", "run_query")
    builder.add_edge("run_query", "generate_final_response")
    builder.add_edge("generate_final_response", END)
    
    return builder.compile()

