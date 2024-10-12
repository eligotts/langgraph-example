from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_mistralai import ChatMistralAI


llm_gpt4o_mini = ChatOpenAI(model="gpt-4o-mini")
llm_gpt35 = ChatOpenAI(model="gpt-3.5-turbo")
llm_claude_haiku = ChatAnthropic(model="claude-3-haiku-20240307")
llm_mistral_small = ChatMistralAI(model="mistral-small-latest")
