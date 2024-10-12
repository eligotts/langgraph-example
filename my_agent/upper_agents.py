from typing import Tuple
import functools
import copy

from my_agent.state import GraphState
from my_agent.models import llm_gpt4o_mini

# Agent that will be giving comments and scores to chains of reasoning
def create_commenter_agent(state, llm):
    """
    Create an agent that comments on the quality of a reasoning chain.

    Args:
        state (dict): The current state containing the question and agent response.
        llm: The language model to use for generating comments.

    Returns:
        dict: Updated state with comments added to the agent response.
    """
    question = state["question"]
    agent_response = state["agent_response"]
    reasoning_chain = agent_response["text"]
    prompt = [
        (
            "system",
            "You are an expert reasoner and excel at assessing the quality of reasoning chains. \n"
            "You will be given a question and a reasoning chain, and you are to assess how well"
            " the reasoning chain does at answering the question both accurately and with sound logic. \n"
            "In response, you will give comments. Your comments should be one to two sentences, and no more than two sentences. \n"
            "Again, in assessing the reasoning chain, focus on both accuracy and "
            "the logical structure of the response. Is the response making invalid assumptions? Could there be a better way to go about answering the question? \n"
            "BE HARSH - think outside the box, just because an answer is "
            "logically sound and organized, doesn't mean it is correct. Think creatively to find shortcomings of the answer, and note these in the comments\n"
            "It is imperative that your response is just the two sentences, with no prefix.",
        ),
        (
            "human",
            f"Here is the question: {question} \n"
            f"Here is the reasoning chain: {reasoning_chain} \n"
        )
    ]
    result = llm.invoke(prompt)
    comments = result.content

    agent_response["comments"] = comments

    return {"agent_response": agent_response}

# Agent that will be giving comments and scores to chains of reasoning
def create_scorer_agent(state, llm):
    """
    Create an agent that scores the quality of a reasoning chain.

    Args:
        state (dict): The current state containing the question, agent response, and comments.
        llm: The language model to use for generating the score.

    Returns:
        dict: Updated state with the score added to the agent response.
    """
    question = state["question"]
    agent_response = state["agent_response"]
    reasoning_chain = agent_response["text"]
    comments = agent_response["comments"]
    prompt = [
        (
            "system",
            "You are an expert reasoner and excel at scoring the quality of a pre-generated answer. \n"
            "You will be given a question, a reasoning chain, and comments on that reasoning chain, and you are to "
            "produce a numeric score of the answer. Respond with a decimal number out of 10, with high numbers representing "
            "high quality responses and low numbers representing low quality responses. BE HARSH - only the finest answers should be "
            "getting the highest marks. \n"
            "It is imperative that your response is just the decimal score out of 10, like 3.5, 5.6, or 7.8.",
        ),
        (
            "human",
            f"Here is the question: {question} \n"
            f"Here is the reasoning chain: {reasoning_chain} \n"
            f"Here are the comments: {comments} \n"
        )
    ]
    result = llm.invoke(prompt)
    score = result.content

    agent_response["score"] = float(score)

    return {"agent_response": agent_response}

# Agent that will be assessing difficulty of the question
def create_difficulty_agent(state, llm):
    """
    Create an agent that assesses the difficulty of a question.

    Args:
        state (dict): The current state containing the question, responses, comments, and grades.
        llm: The language model to use for assessing difficulty.

    Returns:
        dict: A dictionary containing difficulty level and associated parameters.
    """
    question = state["question"]
    responses_full = state["responses"]
    responses = [response["content"][0]["text"] for response in responses_full]
    comments = [response["content"][0]["comments"] for response in responses_full]
    grades = [response["content"][0]["score"] for response in responses_full]

    response1 = responses[0]
    comments1 = comments[0]
    grades1 = grades[0]
    response2 = responses[1]
    comments2 = comments[1]
    grades2 = grades[1]
    response3 = responses[2]
    comments3 = comments[2]
    grades3 = grades[2]
    prompt = [
        (
            "system",
            "You are an expert at chain of thought reasoning and assessing the difficulty of a question. \n"
            "You will be given a question, three different responses to the question, comments on the quality of those responses, and grades on those responses. \n"
            "Using your knowledge of the question, your assessments of the answers, the comments, and grades, you are to assess the difficulty of the question. \n"
            "If all responses are similar with high grades and comments, and you can verify their accuracy, it is probably an easier problem. On the other hand, if the "
            "answers vary widely, that's probably a good sign the question is more challenging. \n"
            "In response, you will simply return a number from 1 to 4, with 1 being an easy question and 4 being a very difficult question. Do not return anything else, just the integer grade on difficulty."
        ),
        (
            "human",
            f"Here is the question: {question} \n"
            f"Here is the first response, comments on the response, and its grade: {response1}, {comments1}, {grades1} \n"
            f"Here is the second response, comments on the response, and its grade: {response2}, {comments2}, {grades2} \n"
            f"Here is the third response, comments on the response, and its grade: {response3}, {comments3}, {grades3} \n",
        )
    ]

    result = llm.invoke(prompt)
    difficulty = result.content

    difficulty_dict = {
        "1": {
            "difficulty": 1,
            "threads": 3,
            "beams": 3,
            "start": 0,
            "revisions": 4
        },
        "2": {
            "difficulty": 2,
            "threads": 5,
            "beams": 3,
            "start": 0,
            "revisions": 3
        },
        "3": {
            "difficulty": 3,
            "threads": 7,
            "beams": 3,
            "start": 0,
            "revisions": 2
        },
        "4": {
            "difficulty": 4,
            "threads": 9,
            "beams": 3,
            "start": 0,
            "revisions": 1
        },
    }
    return difficulty_dict[difficulty]

# Agent that will be checking if the process is done
def create_check_done_agent(state, llm):
    """
    Create an agent that checks if the reasoning process has converged on a correct answer.

    Args:
        state (dict): The current state containing the question and responses.
        llm: The language model to use for checking if the process is done.

    Returns:
        dict: A dictionary indicating whether the process is done.
    """
    question = state["question"]
    responses = state["responses"]

    responses = [response["content"] for response in responses]

    prompt2 = [
        (
            "system",
            "You are an expert at chain of thought reasoning and determining when a correct answer has been converged on. \n"
            "You will be given a question and several reasoning chains containing revised responses to the question, and you are to assess if a correct answer has been converged on, or "
            "if there is still more work to do. \n"
            "Only determine the process is done, and an answer has been "
            "converged on, when there are no more improvements to be made. \n"
            "When you have concluded this, return \"PROCESS DONE\". Otherwise, return \"CONTINUE\". ONLY return one of these two things, and nothing else.",
        ),
        (
            "human",
            f"Here is the question: {question} \n"
            f"Here are the reasoning chains: {str(responses)} \n"
        )
    ]
    result = llm.invoke(prompt2)

    return {"done": result.content == "PROCESS DONE"}

# Agent that will be giving final summary of reasoning chains
def create_final_summary_agent(state, llm):
    """
    Create an agent that generates a final summary of reasoning chains.

    Args:
        state (dict): The current state containing the question and responses.
        llm: The language model to use for generating the final summary.

    Returns:
        dict: A dictionary containing the final combined response.
    """
    question = state["question"]
    responses = state["responses"]

    responses = [response["content"] for response in responses]

    prompt3 = [
        (
            "system",
            "You are an expert at putting complex chains of reasoning into more readable forms. \n"
            "You will be given several reasoning chains in response to a question, with each reasoning chain containing multiple revised responses. \n"
            "FOCUS ONLY ON THE FINAL RESPONSE IN EACH SEPARATE CHAIN, and use the comments and scores associated with each to combine the answers into one, combined answer \n"
            "ONLY return this combined answer, no prefix or anything else"
        ),
        (
            "human",
            f"Here is the question: {question} \n"
            f"Here are the reasoning chains: {str(responses)} \n"
        )
    ]

    result = llm.invoke(prompt3)

    return {"final_response": result.content}

# Create the above agents
commenter_agent = functools.partial(create_commenter_agent, llm=llm_gpt4o_mini)
scorer_agent = functools.partial(create_scorer_agent, llm=llm_gpt4o_mini)
difficulty_agent = functools.partial(create_difficulty_agent, llm=llm_gpt4o_mini)
check_done_agent = functools.partial(create_check_done_agent, llm=llm_gpt4o_mini)
final_summary_agent = functools.partial(create_final_summary_agent, llm=llm_gpt4o_mini)

# Node functions for upper state
def ask_question(state: GraphState) -> GraphState:
    """
    Initialize the state with the question and default values.

    Args:
        state (GraphState): The current state.

    Returns:
        GraphState: The initial state with default values.
    """
    initial_state = {
        "initial_response_agent": "GPT",
        "responses": [],
        "threads": 3,
        "start": 1
    }
    return initial_state

# Passed into response agent chain
def get_info_for_initial_response(state: GraphState) -> Tuple[str, str]:
    """
    Get information for generating the initial response.

    Args:
        state (GraphState): The current state.

    Returns:
        Tuple[str, str]: The question and the initial response agent.
    """
    agent = state["initial_response_agent"]
    return (state["question"], agent)

# Passed into response agent chain
def get_info_for_revision_response(state: GraphState) -> Tuple[str, str, str, str]:
    """
    Get information for generating a revised response.

    Args:
        state (GraphState): The current state.

    Returns:
        Tuple[str, str, str, str]: The question, agent name, current response text, and comments.
    """
    responses = state["responses"]
    index = state["index"]
    cur_response = responses[index]
    return (state["question"], cur_response["agent_name"], cur_response["content"][-1]["text"], cur_response["content"][-1]["comments"])

# Add original response to upper level graph
def join_graph(response: dict):
    """
    Add the original response to the upper level graph.

    Args:
        response (dict): The response to add.

    Returns:
        dict: The agent response containing the final answer.
    """
    return {"agent_response": {"text": response["final_answer"]}}

def beam_search_agent(state: GraphState) -> GraphState:
    """
    Simulate selecting the best responses using beam search.

    Args:
        state (GraphState): The current state.

    Returns:
        GraphState: The updated state with the best responses selected.
    """
    responses = state["responses"]
    responses.sort(key=lambda x: x["content"][-1]["score"], reverse=True)

    beams = state["beams"]
    threads = state["threads"]
    best_responses = responses[:beams]
    bad_responses = responses[beams:]

    final_responses = []
    for response in best_responses:
        for i in range(threads // beams):
            final_responses.append(copy.deepcopy(response))

    for i in range(threads % beams):
        final_responses.append(copy.deepcopy(best_responses[i]))

    return {"responses": final_responses, "discarded_responses": bad_responses, "index": 0}

def initial_response_handler(state: GraphState) -> GraphState:
    """
    Handle the initial response and update the state.

    Args:
        state (GraphState): The current state.

    Returns:
        GraphState: The updated state with the initial response added.
    """
    agent_response = state["agent_response"]
    agent_name = state["initial_response_agent"]
    responses = state["responses"]

    responses.append({"agent_name": agent_name, "content": [agent_response]})

    agent_name_dict = {
        "GPT": "Claude",
        "Claude": "Mistral",
        "Mistral": "GPT"
    }

    return {"responses": responses, "initial_response_agent": agent_name_dict[agent_name]}

def revised_response_handler(state: GraphState) -> GraphState:
    """
    Handle the revised response and update the state.

    Args:
        state (GraphState): The current state.

    Returns:
        GraphState: The updated state with the revised response added.
    """
    agent_response = state["agent_response"]
    index = state["index"]
    responses = state["responses"]

    responses[index]["content"].append(agent_response)

    return {"responses": responses, "index": index + 1}