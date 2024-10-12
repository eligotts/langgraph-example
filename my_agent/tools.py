from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import Tool
from langchain_experimental.utilities import PythonREPL

from langgraph.prebuilt import ToolNode

# Initialize the Tavily search tool with a maximum of 5 results
tavily_tool = TavilySearchResults(max_results=5)

# Initialize a Python REPL (Read-Eval-Print Loop) utility
python_repl = PythonREPL()

# Create a tool for the Python REPL to be used by an agent
repl_tool = Tool(
    name="python_repl",
    description=(
        "A Python shell. Use this to execute python commands, particularly complex math equations. "
        "Input should be a valid python command. If you want to see the output of a value, you should "
        "print it out with `print(...)`. ONLY USE THIS FOR COMPLEX MATH EQUATIONS. For simpler ones "
        "that you can do yourself, you do not need to use this."
    ),
    func=python_repl.run,
)

# List of tools available to the response agents
tools = [tavily_tool, repl_tool]

# Initialize a ToolNode with the list of tools
tool_node = ToolNode(tools)