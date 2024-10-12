from langchain_core.messages import (
    ToolMessage,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage
import functools

from my_agent.tools import tools
from my_agent.models import llm_gpt4o_mini, llm_gpt35, llm_claude_haiku, llm_mistral_small

def create_response_agent(llm, tools):
    """
    Create an agent for generating initial responses to complex questions.

    Args:
        llm: The language model to use for generating responses.
        tools: A list of tools that the agent can use to help answer questions.

    Returns:
        A prompt template bound to the language model and tools.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert reasoner and excel at answering complex questions. \n"
                "You will be given a complex question, and you must use chain of thought reasoning to come to your answer. \n"
                "To help answer your question, you have access to the following tools: {tool_names} \n",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
    return prompt | llm.bind_tools(tools)

def create_summary_agent(llm):
    """
    Create an agent for summarizing complex chains of reasoning.

    Args:
        llm: The language model to use for generating summaries.

    Returns:
        A prompt template bound to the language model.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert at putting complex chains of reasoning into more readable forms. \n"
                "You will be given a reasoning chain in response to a question, and you are to put this chain into a more readable form, clearly describing each step and its output. \n"
                "Here is the reasoning chain:",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    return prompt | llm

def create_revision_agent(llm, tools):
    """
    Create an agent for revising pre-generated answers to questions.

    Args:
        llm: The language model to use for generating revised responses.
        tools: A list of tools that the agent can use to help revise answers.

    Returns:
        A prompt template bound to the language model and tools.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert reasoner and excel at revising pre-generated answers to questions. \n"
                "You will be given a question, a reasoning chain answering the question, and a set of comments "
                "assessing the quality of the reasoning. You must improve upon this answer, using chain of thought reasoning to come to your revised answer. \n"
                "To help, you have access to the following tools: {tool_names} \n"
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
    return prompt | llm.bind_tools(tools)

def agent_node(state, agent, name):
    """
    Helper function to create a node for a given agent.

    Args:
        state: The current state of the conversation.
        agent: The agent to invoke.
        name: The name of the agent.

    Returns:
        A dictionary representing the updated state after invoking the agent.
    """
    result = agent.invoke(state)
    # Convert the agent output into a format that is suitable to append to the global state
    if name == "Summary":
        return {
            "final_answer": result.content,
        }
      
    if isinstance(result, ToolMessage):
        pass
    else:
        result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)
    return {
        "messages": [result],
        # Track the sender to know who to pass to next in the workflow.
        "sender": name,
    }

# Create agents and their corresponding nodes for different models and tasks

# Research agent and node for GPT-4o
gpt_agent = create_response_agent(llm_gpt4o_mini, tools)
gpt_node = functools.partial(agent_node, agent=gpt_agent, name="GPT")

# Research agent and node for Claude Haiku
claude_agent = create_response_agent(llm_claude_haiku, tools)
claude_node = functools.partial(agent_node, agent=claude_agent, name="Claude")

# Research agent and node for Mistral Small
mistral_agent = create_response_agent(llm_mistral_small, tools)
mistral_node = functools.partial(agent_node, agent=mistral_agent, name="Mistral")

# Summary agent and node for GPT-4o Mini
summary_agent = create_summary_agent(llm_gpt4o_mini)
summary_node = functools.partial(agent_node, agent=summary_agent, name="Summary")

# Revision agent and node for GPT-4o
gpt_revision = create_revision_agent(llm_gpt4o_mini, tools)
gpt_revision_node = functools.partial(agent_node, agent=gpt_revision, name="GPT")

# Revision agent and node for Claude Haiku
claude_revision = create_revision_agent(llm_claude_haiku, tools)
claude_revision_node = functools.partial(agent_node, agent=claude_revision, name="Claude")

# Revision agent and node for Mistral Small
mistral_revision = create_revision_agent(llm_mistral_small, tools)
mistral_revision_node = functools.partial(agent_node, agent=mistral_revision, name="Mistral")