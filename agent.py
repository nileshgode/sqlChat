# app/agent.py

import logging
from typing import TypedDict, Annotated, List
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.graph import StateGraph, END, START

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- 1. Define State ---
class AgentState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]
    user_question: str
    schema: str
    sql_query: str
    query_result: str
    final_answer: str


# --- Node Functions ---
def get_schema_node(state: AgentState):
    """Fetch database schema and store in state."""
    logger.info("Node: get_schema")
    return {"schema": state["schema"], "messages": [SystemMessage(content=state["schema"])]}


def call_model_to_generate_query(state: AgentState, llm):
    """Generates the SQL query using LLM."""
    logger.info("Node: call_model_to_generate_query")
    prompt = f"""
You are a SQL expert. Based on the user's question and the database schema,
generate ONLY a syntactically correct SQL query.

User Question:
{state['user_question']}

Schema:
{state['schema']}

Output ONLY the SQL query, no explanation, no commentary.
"""
    response = llm.invoke(prompt)

    # Normalize response to string
    if isinstance(response, str):
        sql = response.strip().strip("`")
    else:
        sql = response.content.strip().strip("`")

    logger.info(f"Generated SQL: {sql}")
    return {"sql_query": sql}


def execute_sql_query(state: AgentState, run_query_tool):
    """Executes the generated SQL query using the DB tool."""
    logger.info("Node: execute_sql_query")
    query = state["sql_query"]
    try:
        result = run_query_tool.invoke({"query": query})
        logger.info(f"SQL Result: {result}")
        return {"query_result": str(result)}
    except Exception as e:
        err = f"SQL execution failed: {e}"
        logger.error(err)
        return {"query_result": err}


def summarize_result(state: AgentState, llm):
    """Summarizes the SQL query result into a final answer."""
    logger.info("Node: summarize_result")
    query_result_str = str(state.get('query_result', ''))

    prompt = f"""
You are a helpful assistant. Based on the user's question and the result of a database query, provide a clear, natural language answer.

User Question:
{state.get('user_question', '')}

Database Query Result:
{query_result_str}

Final Answer:
"""
    response = llm.invoke(prompt)

    # Normalize response to string
    if isinstance(response, str):
        answer = response.strip()
    else:
        answer = response.content.strip()

    logger.info(f"Final Answer: {answer}")

    return {
        "messages": [AIMessage(content=answer)],
        "final_answer": answer
    }


# --- Graph Builder ---
def create_sql_agent_graph(llm_instance, db):
    """Builds the SQL agent LangGraph with schema, query generation, execution, summarization."""
    toolkit = SQLDatabaseToolkit(db=db, llm=llm_instance)
    tools = toolkit.get_tools()

    run_query_tool = next(t for t in tools if t.name == "sql_db_query")
    schema_tool = next(t for t in tools if t.name == "sql_db_schema")

    # fetch schema once at init
    table_names = db.get_usable_table_names()
    schema_description = schema_tool.invoke({"table_names": ", ".join(table_names)})

    builder = StateGraph(AgentState)

    # Add nodes
    builder.add_node("get_schema", lambda state: get_schema_node({**state, "schema": schema_description}))
    builder.add_node("generate_query", lambda state: call_model_to_generate_query(state, llm_instance))
    builder.add_node("execute_query", lambda state: execute_sql_query(state, run_query_tool))
    builder.add_node("summarize_result", lambda state: summarize_result(state, llm_instance))

    # Add edges
    builder.add_edge(START, "get_schema")
    builder.add_edge("get_schema", "generate_query")
    builder.add_edge("generate_query", "execute_query")
    builder.add_edge("execute_query", "summarize_result")
    builder.add_edge("summarize_result", END)

    return builder.compile()
