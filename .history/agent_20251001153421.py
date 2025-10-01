# app/agent.py

from typing import Literal
from langchain_core.messages import AIMessage, BaseMessage
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langgraph.graph import MessagesState # Correct import for MessagesState

# This function is new
def generate_final_response(state: MessagesState):
    """
    Generates a final, natural language response to the user based on the conversation history.
    """
    system_prompt = (
        "You are a helpful assistant. Based on the conversation history, "
        "provide a clear and concise answer to the user's original question. "
        "Summarize the findings from any tool outputs."
    )
    
    # The last message is the output from the `run_query` tool. We use it as context.
    context = state["messages"][-1].content
    
    # The original user question is typically the first message.
    user_question = state["messages"][0].content
    
    prompt = (
        f"{system_prompt}\n\n"
        f"Conversation History:\n"
        f"User: {user_question}\n"
        f"Tool Output: {context}\n\n"
        f"Please provide the final answer."
    )
    
    # We don't need tools for the final answer, just the LLM.
    response = llm.invoke(prompt)
    return {"messages": [response]}


def create_sql_agent_graph(llm_instance, db):
    """
    Creates and compiles a custom LangGraph SQL agent with a dedicated final response node.
    """
    global llm # Make llm accessible to the new function
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

    generate_query_system_prompt = f"..." # Same as before
    def generate_query(state: MessagesState):
        # ... same as before
        system_message = {"role": "system", "content": generate_query_system_prompt}
        llm_with_tools = llm.bind_tools([run_query_tool])
        response = llm_with_tools.invoke([system_message] + state["messages"])
        return {"messages": [response]}

    check_query_system_prompt = f"..." # Same as before
    def check_query(state: MessagesState):
        # ... same as before
        system_message = {"role": "system", "content": check_query_system_prompt}
        tool_call = state["messages"][-1].tool_calls[0]
        user_message = {"role": "user", "content": tool_call["args"]["query"]}
        llm_with_tools = llm.bind_tools([run_query_tool], tool_choice="any")
        response = llm_with_tools.invoke([system_message, user_message])
        response.id = state["messages"][-1].id
        return {"messages": [response]}

    # This conditional edge is now simpler
    def should_continue(state: MessagesState) -> Literal["check_query", "generate_final_response"]:
        # If the last message has tool calls, continue to check query.
        # Otherwise, the model has likely produced the query result, so generate the final answer.
        if state["messages"][-1].tool_calls:
            return "check_query"
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
    builder.add_node("generate_final_response", generate_final_response) # Add the new node

    builder.add_edge(START, "list_tables")
    builder.add_edge("list_tables", "call_get_schema")
    builder.add_edge("call_get_schema", "get_schema")
    builder.add_edge("get_schema", "generate_query")
    
    # This logic has changed
    builder.add_conditional_edges("generate_query", should_continue)
    
    builder.add_edge("check_query", "run_query")
    builder.add_edge("run_query", "generate_final_response") # Go to final response after running query
    builder.add_edge("generate_final_response", END) # The new final node ends the graph
    
    return builder.compile()
