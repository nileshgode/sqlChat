# app/agent.py

import logging
from typing import Literal
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
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

# --- THIS IS THE CRITICAL AND FINAL FIX ---
def generate_final_response(state: MessagesState):
    logger.info("Entering generate_final_response node")
    
    # Get the original user question
    user_question = state["messages"][0].content
    
    # Find the last ToolMessage, which contains the result of the SQL query
    tool_output = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, ToolMessage):
            tool_output = msg.content
            break

    if not tool_output:
        # This is a safeguard, but with the current graph, it shouldn't be hit.
        return {"messages": [AIMessage(content="I was unable to get a result from the database.")]}

    # Create a very specific prompt for the final answer.
    system_prompt = f"""You are a helpful assistant. Your task is to provide a final, natural language answer to the user's question based *only* on the provided tool output.

    Original Question: {user_question}

    Tool Output (Database Query Result):
    {tool_output}

    Based on the tool output, what is the answer to the user's question?
    """
    
    # We invoke the LLM with just this specific prompt, not the whole history.
    response = llm.invoke(system_prompt)
    logger.info(f"Final response generated: {response.content}")
    return {"messages": [response]}


def list_tables(state: MessagesState):
    # (No changes from previous version)
    logger.info("Entering list_tables node")
    tool_call = {"name": "sql_db_list_tables", "args": {}, "id": "abc123", "type": "tool_call"}
    tool_call_message = AIMessage(content="", tool_calls=[tool_call])
    tool_message = list_tables_tool.invoke(tool_call)
    response = AIMessage(f"Available tables: {tool_message.content}")
    return {"messages": [tool_call_message, tool_message, response]}

def call_get_schema(state: MessagesState):
    # (No changes from previous version)
    logger.info("Entering call_get_schema node")
    llm_with_tools = llm.bind_tools([get_schema_tool], tool_choice="any")
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

def generate_query(state: MessagesState):
    # (No changes from previous version)
    logger.info("Entering generate_query node")
    generate_query_system_prompt = "You are an agent designed to interact with a SQL database. Given an input question, your primary job is to generate a syntactically correct SQL query. Output ONLY the SQL query."
    system_message = {"role": "system", "content": generate_query_system_prompt}
    response = llm.invoke([system_message] + state["messages"])
    logger.info(f"LLM output from generate_query: {response.content}")
    return {"messages": [response]}

def check_query(state: MessagesState):
    # (No changes from previous version)
    logger.info("Entering check_query node")
    last_message = state["messages"][-1]
    query_text = last_message.content

    force_run_prompt = f"You are a query executor. Your only job is to take the following SQL query and execute it using the available tool. Do not modify it.\n\nQuery to execute:\n{query_text}"
    llm_with_forced_tool = llm.bind_tools([run_query_tool], tool_choice="required")
    response = llm_with_forced_tool.invoke(force_run_prompt)
    return {"messages": [response]}


# --- Graph Builder Function ---
def create_sql_agent_graph(llm_instance, db):
    # (No changes from previous version)
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

