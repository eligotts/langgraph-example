# Test-Time Compute Simulation

Welcome to the Test-Time Compute Simulation Tool! This LangGraph workflow allows you to interact with AI agents to answer complex questions, giving you freedom to alter the specifics of your test-time compute algorithm. This project was inspired by this research paper: https://arxiv.org/pdf/2408.03314.

For the command line tool version of this project, see here: https://github.com/eligotts/test-time-compute

For a Google Colab version of this project, see here: https://colab.research.google.com/drive/11exuBZEPr0ITRM12G5V7r_u9mcWuBwzE?usp=sharing

## Theory
This application is a simplification of the procedure described in the paper linked above. Here is a high level overview:

- **Initial Response Generation**: first, we generate a certain amount of responses to the question.
- **Comments and Scores**: Each response generated receives comments on its accuracy and logical structure. It also receives a numeric grade out of 10 on the quality of the response.
- **Difficulty Assessment**: after the initial responses have been commented on and scored, we assess how difficult the question.
- **Best-of-N vs. Revisions vs. Beam Search**: given the difficulty, we need to distribute our compute resources. We can generate more individual threads (best-of-N), or generate fewer individual threads, and revise more. After each revision step, we can also choose to discard our worst responses (beam search). See `create_difficulty_agent` in `my_agent/upper_agents.py` to edit this compute resource distribution.
- **Final Summary**: Once we decide we are done, or are out of compute, we generate a final, coherent answer to the user.


## Installation

### Prerequisites

- **Python**: Ensure you have Python 3.7 or higher installed. [Download Python](https://www.python.org/downloads/)

- **LangGraph Studio**: Download LangGraph Studio to run this workflow. Will need Docker as well.

### Steps

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/eligotts/test-time-compute.git
   ```

2. **Navigate to the Project Directory**:

   ```bash
   cd test-time-compute
   ```

4. **Add your API keys**:

   Create a `.env` file in the root directory, and enter your API keys

   ```
   OPENAI_API_KEY=your_openai_api_key
   ANTHROPIC_API_KEY=your_anthropic_api_key
   MISTRAL_API_KEY=your_mistral_api_key
   TAVILY_API_KEY=your_tavily_api_key
   ```

5. **Load the flow in LangGraph Studio**: 

    Open LangGraph Studio, and select project directory!

6. **Input your question**:

    Enter your question into the `question` field, and hit `Submit`. Watch the reasoning steps go...

### Please edit and add stuff if you're interested! 
