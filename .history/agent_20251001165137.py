# app/agent.py

from typing import Literal
from langchain_core.messages import AIMessage, BaseMessage
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langgraph.graph import MessagesState

# This function is now simpler and more robust
def generate_final_response(state: MessagesState):
    """
    Generates a final, natural language response based on the full conversation history.
    """
    # The entire history is already in state["messages"]
    # We just need to prompt the LLM to summarize it.
    system_prompt = (
        "You are a helpful assistant. Based on the entire conversation history, "
        "provide a clear and concise final answer to the user's original question. "
        "Synthesize the information from any tool outputs into a natural language response."
    )
    
    # We create a new list of messages for the final prompt
    final_prompt_messages = [
        {"role": "system", "content": system_prompt},
    ] + state["messages"]

    # We don't need tools for the final answer, just the LLM.
    response = llm.invoke(final_prompt_messages)
    return {"messages": [response]}


def create_sql_agent_graph(llm_instance, db):
    """
    Creates and compiles a custom LangGraph SQL agent.
    """
    global llm
    llm = llm_instance

    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()

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

    generate_query_system_prompt = f"..." # No change
    def generate_query(state: MessagesState):
        system_message = {"role": "system", "content": generate_query_system_prompt}
        llm_with_tools = llm.bind_tools([run_query_tool])
        response = llm_with_tools.invoke([system_message] + state["messages"])
        return {"messages": [response]}

    check_query_system_prompt = f"..." # No change
    def check_query(state: MessagesState):
        system_message = {"role": "system", "content": check_query_system_prompt}
        tool_call = state["messages"][-1].tool_calls[0]
        user_message = {"role": "user", "content": tool_call["args"]["query"]}
        llm_with_tools = llm.bind_tools([run_query_tool], tool_choice="any")
        response = llm_with_tools.invoke([system_message, user_message])
        response.id = state["messages"][-1].id
        return {"messages": [response]}

    def should_continue(state: MessagesState) -> Literal["check_query", "generate_final_response"]:
        # The logic here is critical.
        # If the last message is from the AI and contains a tool call, we need to execute it.
        if isinstance(state['messages'][-1], AIMessage) and state['messages'][-1].tool_calls:
            return "check_query"
        # Otherwise, we have the tool output and can generate the final response.
        else:
            return "generate_final_response"

    # Build the Graph
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
    builder.add_conditional_edges("generate_query", should_continue)
    builder.add_edge("check_query", "run_query")
    builder.add_edge("run_query", "generate_final_response")
    builder.add_edge("generate_final_response", END)
    
    return builder.compile()

