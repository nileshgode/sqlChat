# app/agent.py

from typing import Literal
from langchain_core.messages import AIMessage
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode, MessagesState

def create_sql_agent_graph(llm, db):
    """
    Creates and compiles a custom LangGraph SQL agent.

    Args:
        llm: The language model instance.
        db: The SQLDatabase instance.

    Returns:
        A compiled LangGraph agent.
    """
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()

    # Define Nodes
    get_schema_tool = next(tool for tool in tools if tool.name == "sql_db_schema")
    run_query_tool = next(tool for tool in tools if tool.name == "sql_db_query")

    def list_tables(state: MessagesState):
        list_tables_tool = next(tool for tool in tools if tool.name == "sql_db_list_tables")
        tool_call = {"name": "sql_db_list_tables", "args": {}, "id": "abc123", "type": "tool_call"}
        tool_call_message = AIMessage(content="", tool_calls=[tool_call])
        tool_message = list_tables_tool.invoke(tool_call)
        response = AIMessage(f"Available tables: {tool_message.content}")
        return {"messages": [tool_call_message, tool_message, response]}

    def call_get_schema(state: MessagesState):
        llm_with_tools = llm.bind_tools([get_schema_tool], tool_choice="any")
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    generate_query_system_prompt = f"""
    You are an agent designed to interact with a SQL database with the dialect '{db.dialect}'.
    Given an input question, create a syntactically correct query to run.
    Unless the user specifies a number of examples, limit your query to at most 5 results.
    Never query for all columns; only ask for the relevant columns.
    DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.).
    """
    def generate_query(state: MessagesState):
        system_message = {"role": "system", "content": generate_query_system_prompt}
        llm_with_tools = llm.bind_tools([run_query_tool])
        response = llm_with_tools.invoke([system_message] + state["messages"])
        return {"messages": [response]}

    check_query_system_prompt = f"""
    You are a SQL expert. Double check the given query for common mistakes.
    If there are mistakes, rewrite it. If not, reproduce the original query.
    You will call the appropriate tool to execute the query after this check.
    """
    def check_query(state: MessagesState):
        system_message = {"role": "system", "content": check_query_system_prompt}
        tool_call = state["messages"][-1].tool_calls[0]
        user_message = {"role": "user", "content": tool_call["args"]["query"]}
        llm_with_tools = llm.bind_tools([run_query_tool], tool_choice="any")
        response = llm_with_tools.invoke([system_message, user_message])
        response.id = state["messages"][-1].id
        return {"messages": [response]}

    def should_continue(state: MessagesState) -> Literal[END, "check_query"]:
        return "check_query" if state["messages"][-1].tool_calls else END

    # Build Graph
    builder = StateGraph(MessagesState)
    builder.add_node("list_tables", list_tables)
    builder.add_node("call_get_schema", call_get_schema)
    builder.add_node("get_schema", ToolNode([get_schema_tool]))
    builder.add_node("generate_query", generate_query)
    builder.add_node("check_query", check_query)
    builder.add_node("run_query", ToolNode([run_query_tool]))

    builder.add_edge(START, "list_tables")
    builder.add_edge("list_tables", "call_get_schema")
    builder.add_edge("call_get_schema", "get_schema")
    builder.add_edge("get_schema", "generate_query")
    builder.add_conditional_edges("generate_query", should_continue)
    builder.add_edge("check_query", "run_query")
    builder.add_edge("run_query", "generate_query")
    
    return builder.compile()
