from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
import os
from src.MAS_supervisor.tools.math_tools import *
from src.MAS_supervisor.utils.pretty_print import pretty_print_message, pretty_print_messages

os.environ["OPENAI_API_KEY"] = "tpsg-QS1lyFgvVj8qu21PlxfFaZLtAtNxpY3"

llm = ChatOpenAI(
    openai_api_key="tpsg-QS1lyFgvVj8qu21PlxfFaZLtAtNxpY3",
    base_url="https://api.metisai.ir/openai/v1",
    model="gpt-4.1-mini", 
    temperature=0
)


def math_agent(llm=llm):
    return create_react_agent(
        model=llm,
        tools=[add, multiply, divide],
        prompt=(
            "You are a math agent.\n\n"
            "INSTRUCTIONS:\n"
            "- Assist ONLY with math-related tasks\n"
            "- After you're done with your tasks, respond to the supervisor directly\n"
            "- Respond ONLY with the results of your work, do NOT include ANY other text."
        ),
        name="math_agent",
    )

    # for chunk in math_agent.stream(
    #     {"messages": [{"role": "user", "content": "what's (3 + 4) / 7"}]}
    # ):
    #     pretty_print_messages(chunk)