import os
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent
from langchain_cohere import ChatCohere
# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI

# from ...utils.pretty_print import pretty_print_message, pretty_print_messages
from src.MAS_supervisor.utils.pretty_print import pretty_print_message, pretty_print_messages

# exit(0)
from langchain_core.messages import convert_to_messages

os.environ["TAVILY_API_KEY"] = "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"
os.environ["OPENAI_API_KEY"] = "tpsg-QS1lyFgvVj8qu21PlxfFaZLtAtNxpY3"

web_search = TavilySearch(max_results=3)#, api_key="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")
# web_search_results = web_search.invoke("who is the mayor of NYC?")

# print(web_search_results["results"][0]["content"])


llm = ChatOpenAI(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        base_url="https://api.metisai.ir/openai/v1",
        model="gpt-4.1-mini", 
        temperature=0)

# print(llm.invoke("Hello, my name is Inigo Montoya"))

def research_agent(llm=llm):
    return create_react_agent(
        model=llm,
        tools=[web_search],
        prompt=(
            "You are a research agent.\n\n"
            "INSTRUCTIONS:\n"
            "- Assist ONLY with research-related tasks, DO NOT do any math\n"
            "- After you're done with your tasks, respond to the supervisor directly\n"
            "- Respond ONLY with the results of your work, do NOT include ANY other text."
        ),
        name="research_agent",
    )


    # for chunk in research_agent.stream(
    #     {"messages": [{"role": "user", "content": "who is the mayor of NYC?"}]}
    # ):
    #     # print(chunk, end="", flush=True)
    #     pretty_print_messages(chunk)

    