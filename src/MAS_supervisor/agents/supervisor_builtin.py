from langgraph_supervisor import create_supervisor
from langchain_openai import ChatOpenAI
import os
from src.MAS_supervisor.agents.math_agent import math_agent
from src.MAS_supervisor.agents.research_agent import research_agent
from src.MAS_supervisor.utils.pretty_print import pretty_print_message, pretty_print_messages

os.environ["OPENAI_API_KEY"] = "tpsg-QS1lyFgvVj8qu21PlxfFaZLtAtNxpY3"

llm = ChatOpenAI(
    openai_api_key="tpsg-QS1lyFgvVj8qu21PlxfFaZLtAtNxpY3",
    base_url="https://api.metisai.ir/openai/v1",
    model="gpt-4.1-mini", 
    temperature=0
)

supervisor = create_supervisor(
    model=llm,
    agents=[math_agent(llm), research_agent(llm)],
    prompt=(
        "You are a supervisor managing two agents:\n"
        "- a research agent. Assign research-related tasks to this agent\n"
        "- a math agent. Assign math-related tasks to this agent\n"
        "Assign work to one agent at a time, do not call agents in parallel.\n"
        "If one agent needs the output of another agent, first assign the task to the first agent, then assign a task to the second agent using the output of the first agent.\n"
        "If a task is complex and involves both research and math, break it down into smaller tasks and assign them to the appropriate agents.\n"
        "Do not do any work yourself."
    ),
    name="supervisor",
    add_handoff_back_messages=True,
    output_mode="full_history",
).compile()

# for chunk in supervisor.stream(
#     {"messages": [{"role": "user", "content": "What is the height of Mount Everest divided by the number of countries in the world?"}]}
# ):
#     pretty_print_messages(chunk)


from IPython.display import display, Image
from PIL import Image
import io

img_bytes = supervisor.get_graph().draw_mermaid_png()
lang_graph_img = Image.open(io.BytesIO(img_bytes))
lang_graph_img.save("lang_graph.png")
# display(lang_graph_img)