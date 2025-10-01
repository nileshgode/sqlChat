# app/agent.py

import logging
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage, AIMessage
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.graph import StateGraph, END, START

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 1. Define a Custom, Structured State ---
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]
    sql_query: str
    query_result: str

# --- Module-level variables ---
llm = None
run_query_tool = None

# --- Node Functions (Updated for Custom State) ---

def call_model_to_generate_query(state: AgentState):
    """Generates the SQL query."""
    logger.info("Node: call_model_to_generate_query")
    prompt = f"""You are a SQL expert. Based on the user's question and the database schema, generate a single, syntactically correct SQL query.

    User Question:
    {state['messages'][-1].content}

    Schema:
    {state['messages'][-2].content}

    Output ONLY the SQL query.
    """
    response = llm.invoke(prompt)
    logger.info(f"Generated SQL: {response.content}")
    return {"sql_query": response.content}

def execute_sql_query(state: AgentState):
    """Executes the generated SQL query."""
    logger.info("Node: execute_sql_query")
    query = state["sql_query"]
    result = run_query_tool.invoke({"query": query})
    logger.info(f"SQL Result: {result}")
    return {"query_result": result}

# --- THIS IS THE CRITICAL AND FINAL FIX ---
def summarize_result(state: AgentState):
    """Summarizes the query result into a final answer."""
    logger.info("Node: summarize_result")
    
    # Convert the query result (which might be a list of tuples) to a clean string
    query_result_str = str(state['query_result'])

    prompt = f"""You are a helpful assistant. Based on the user's question and the result of a database query, provide a clear, natural language answer.

    User Question:
    {state['messages'][0].content}

    Database Query Result:
    {query_result_str}

    Final Answer:
    """
    response = llm.invoke(prompt)
    logger.info(f"Final Answer: {response.content}")
    # Return the response as a valid BaseMessage object
    return {"messages": [AIMessage(content=response.content)]}


# --- Graph Builder Function ---
def create_sql_agent_graph(llm_instance, db):
    """Initializes tools and compiles the structured LangGraph SQL agent."""
    global llm, run_query_tool
    llm = llm_instance

    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()
    run_query_tool = next(tool for tool in tools if tool.name == "sql_db_query")
    
    schema_tool = next(tool for tool in tools if tool.name == "sql_db_schema")
    table_names = db.get_usable_table_names()
    schema_description = schema_tool.invoke({"table_names": ", ".join(table_names)})
    
    builder = StateGraph(AgentState)

    builder.add_node("get_schema", lambda state: {"messages": [("system", schema_description)]})
    builder.add_node("generate_query", call_model_to_generate_query)
    builder.add_node("execute_query", execute_sql_query)
    builder.add_node("summarize_result", summarize_result)

    builder.add_edge(START, "get_schema")
    builder.add_edge("get_schema", "generate_query")
    builder.add_edge("generate_query", "execute_query")
    builder.add_edge("execute_query", "summarize_result")
    builder.add_edge("summarize_result", END)
    
    return builder.compile()

