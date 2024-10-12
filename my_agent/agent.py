from typing import Literal
from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph, START

from my_agent.state import AgentState, GraphState
from my_agent.response_agents import (
    gpt_node, claude_node, mistral_node, summary_node, 
    gpt_revision_node, claude_revision_node, mistral_revision_node
)
from my_agent.tools import tool_node
from my_agent.upper_agents import (
    ask_question, join_graph, get_info_for_initial_response, 
    get_info_for_revision_response, difficulty_agent, commenter_agent, 
    scorer_agent, check_done_agent, final_summary_agent, beam_search_agent, 
    initial_response_handler, revised_response_handler
)

# ROUTERS
def router_tools(state) -> Literal["call_tool", "__end__"]:
    """
    Router function to determine if a tool should be called or if the process should end.
    
    Args:
        state (dict): The current state of the agent.
        
    Returns:
        Literal["call_tool", "__end__"]: The next step in the workflow.
    """
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        # The previous agent is invoking a tool
        return "call_tool"
    return "__end__"

def initial_response_router(state) -> Literal["get_initial_response", "difficulty_assessment", "beam_search_agent"]:
    """
    Router function for initial response workflow.
    
    Args:
        state (dict): The current state of the agent.
        
    Returns:
        Literal["get_initial_response", "difficulty_assessment", "beam_search_agent"]: The next step in the workflow.
    """
    start = state["start"]
    responses = state["responses"]
    threads = state["threads"]

    if len(responses) < threads:
        return "get_initial_response"
    elif bool(int(start)):
        return "difficulty_assessment"

    return "beam_search_agent"

def difficulty_router(state) -> Literal["get_initial_response", "beam_search_agent"]:
    """
    Router function for difficulty assessment workflow.
    
    Args:
        state (dict): The current state of the agent.
        
    Returns:
        Literal["get_initial_response", "beam_search_agent"]: The next step in the workflow.
    """
    responses = state["responses"]
    threads = state["threads"]

    if len(responses) < threads:
        return "get_initial_response"
    
    return "beam_search_agent"

def revision_router(state) -> Literal["check_done", "get_revision_response", "summary"]:
    """
    Router function for revision workflow.
    
    Args:
        state (dict): The current state of the agent.
        
    Returns:
        Literal["check_done", "get_revision_response", "summary"]: The next step in the workflow.
    """
    index = state["index"]
    threads = state["threads"]
    revisions = state["revisions"]
    responses = state["responses"]
    if len(responses[-1]["content"]) == revisions + 1:
        return "summary"
    elif index == threads:
        return "check_done"

    return "get_revision_response"

def scorer_router(state) -> Literal["initial_response_handler", "revised_response_handler"]:
    """
    Router function for scorer workflow.
    
    Args:
        state (dict): The current state of the agent.
        
    Returns:
        Literal["initial_response_handler", "revised_response_handler"]: The next step in the workflow.
    """
    start = state["start"]
    threads = state["threads"]
    responses = state["responses"]

    if bool(int(start)) or len(responses) < threads:
        return "initial_response_handler"
    
    return "revised_response_handler"

def done_router(state) -> Literal["continue", "__end__"]:
    """
    Router function to determine if the process is done or should continue.
    
    Args:
        state (dict): The current state of the agent.
        
    Returns:
        Literal["continue", "__end__"]: The next step in the workflow.
    """
    if state["done"]:
        return "__end__"
    else:
        return "continue"

# CREATE THE INITIAL RESPONSE GRAPH
initial_response_workflow = StateGraph(AgentState)

# Add nodes to the initial response workflow
initial_response_workflow.add_node("GPT", gpt_node)
initial_response_workflow.add_node("Claude", claude_node)
initial_response_workflow.add_node("Mistral", mistral_node)
initial_response_workflow.add_node("call_tool", tool_node)
initial_response_workflow.add_node("Summary", summary_node)

# Add conditional edges to the initial response workflow
initial_response_workflow.add_conditional_edges(
    "GPT",
    router_tools,
    {"call_tool": "call_tool", "__end__": "Summary"},
)
initial_response_workflow.add_conditional_edges(
    "Claude",
    router_tools,
    {"call_tool": "call_tool", "__end__": "Summary"},
)
initial_response_workflow.add_conditional_edges(
    "Mistral",
    router_tools,
    {"call_tool": "call_tool", "__end__": "Summary"},
)

initial_response_workflow.add_conditional_edges(
    "call_tool",
    # Each agent node updates the 'sender' field
    # the tool calling node does not, meaning
    # this edge will route back to the original agent
    # who invoked the tool
    lambda x: x["sender"],
    {
        "GPT": "GPT",
        "Claude": "Claude",
        "Mistral": "Mistral",
    },
)

initial_response_workflow.add_conditional_edges(
    START, 
    # The upper graph tells the initial response agent
    # which agent to use in the response, so we want 
    # to go to that agent to generate an answer
    lambda x: x["sender"],
    {
        "GPT": "GPT",
        "Claude": "Claude",
        "Mistral": "Mistral",
    },
)

initial_response_workflow.add_edge("Summary", END)
graph_initial = initial_response_workflow.compile()

def enter_chain(message_and_agent):
    """
    Function to kick off the initial response chain.
    
    Args:
        message_and_agent (tuple): A tuple containing the message and the agent.
        
    Returns:
        dict: The initial state for the workflow.
    """
    message, agent = message_and_agent
    results = {
        "messages": [HumanMessage(content=message)],
        "sender": agent,
    }
    return results

initial_response_chain = enter_chain | graph_initial

# CREATE THE REVISION GRAPH
revision_workflow = StateGraph(AgentState)

# Add nodes to the revision workflow
revision_workflow.add_node("GPT", gpt_revision_node)
revision_workflow.add_node("Claude", claude_revision_node)
revision_workflow.add_node("Mistral", mistral_revision_node)
revision_workflow.add_node("call_tool", tool_node)
revision_workflow.add_node("Summary", summary_node)

# Add conditional edges to the revision workflow
revision_workflow.add_conditional_edges(
    "GPT",
    router_tools,
    {"call_tool": "call_tool", "__end__": "Summary"},
)
revision_workflow.add_conditional_edges(
    "Claude",
    router_tools,
    {"call_tool": "call_tool", "__end__": "Summary"},
)
revision_workflow.add_conditional_edges(
    "Mistral",
    router_tools,
    {"call_tool": "call_tool", "__end__": "Summary"},
)

revision_workflow.add_conditional_edges(
    "call_tool",
    # Each agent node updates the 'sender' field
    # the tool calling node does not, meaning
    # this edge will route back to the original agent
    # who invoked the tool
    lambda x: x["sender"],
    {
        "GPT": "GPT",
        "Claude": "Claude",
        "Mistral": "Mistral",
    },
)
revision_workflow.add_conditional_edges(
    START, 
    # The upper graph tells the initial response agent
    # which agent to use in the response, so we want 
    # to go to that agent to generate an answer
    lambda x: x["sender"],
    {
        "GPT": "GPT",
        "Claude": "Claude",
        "Mistral": "Mistral",
    },
)
revision_workflow.add_edge("Summary", END)
graph_revision = revision_workflow.compile()

def enter_chain_revision(question_and_agent_and_previous_response_and_comments):
    """
    Function to kick off revision chain.
    
    Args:
        question_and_agent_and_previous_response_and_comments (tuple): A tuple containing the question, agent, previous response, and comments.
        
    Returns:
        dict: The initial state for the workflow.
    """
    question, agent, previous_response, comments = question_and_agent_and_previous_response_and_comments
    message = "Here is the question: " + question + "\n"
    message += "Here is the previous response: " + previous_response + "\n"
    message += "Here are the comments: " + comments + "\n"
    results = {
        "messages": [HumanMessage(content=message)],
        "sender": agent,
    }
    return results

revision_chain = enter_chain_revision | graph_revision

# Build the main graph
graph = StateGraph(GraphState)

# Add nodes to the main graph
graph.add_node("ask_question", ask_question)
graph.add_node("get_initial_response", get_info_for_initial_response | initial_response_chain | join_graph)
graph.add_node("get_revision_response", get_info_for_revision_response | revision_chain | join_graph)
graph.add_node("difficulty_assessment", difficulty_agent)
graph.add_node("commenter", commenter_agent)
graph.add_node("scorer", scorer_agent)
graph.add_node("check_done", check_done_agent)
graph.add_node("final_summary", final_summary_agent)
graph.add_node("beam_search_agent", beam_search_agent)
graph.add_node("initial_response_handler", initial_response_handler)
graph.add_node("revised_response_handler", revised_response_handler)

# Define edges in the main graph
graph.set_entry_point("ask_question")
graph.add_edge("ask_question", "get_initial_response")
graph.add_edge("get_initial_response", "commenter")
graph.add_edge("commenter", "scorer")
graph.add_conditional_edges(
    "scorer",
    scorer_router,
    {"initial_response_handler": "initial_response_handler", "revised_response_handler": "revised_response_handler"},
)

graph.add_conditional_edges(
    "initial_response_handler",
    initial_response_router,
    {"get_initial_response": "get_initial_response", "difficulty_assessment": "difficulty_assessment", "beam_search_agent": "beam_search_agent"},
)

graph.add_conditional_edges(
    "difficulty_assessment",
    difficulty_router,
    {"get_initial_response": "get_initial_response", "beam_search_agent": "beam_search_agent"},
)

graph.add_edge("beam_search_agent", "get_revision_response")
graph.add_edge("get_revision_response", "commenter")

graph.add_conditional_edges(
    "revised_response_handler",
    revision_router,
    {"get_revision_response": "get_revision_response", "check_done": "check_done", "summary": "final_summary"},
)

graph.add_conditional_edges(
    "check_done",
    done_router,
    {"continue": "beam_search_agent", "__end__": "final_summary"},
)
graph.add_edge("final_summary", END)

# Compile the main graph
app = graph.compile()