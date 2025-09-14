from langgraph.graph import StateGraph, MessagesState, START, END
from src.MAS_hierarichal.agents.research_agent_team import *
from src.MAS_hierarichal.agents.document_writing_team import *

def research_graph():
    research_builder = StateGraph(State)
    research_builder.add_node("supervisor", research_supervisor_node)
    research_builder.add_node("search", search_node)
    research_builder.add_node("web_scraper", web_scraper_node)

    research_builder.add_edge(START, "supervisor")
    research_graph = research_builder.compile()

    return research_graph

def paper_writing_graph():
    paper_writing_builder = StateGraph(State)
    paper_writing_builder.add_node("supervisor", doc_writing_supervisor_node)
    paper_writing_builder.add_node("doc_writer", doc_writing_node)
    paper_writing_builder.add_node("note_taker", note_taking_node)
    paper_writing_builder.add_node("chart_generator", chart_generating_node)

    paper_writing_builder.add_edge(START, "supervisor")
    paper_writing_graph = paper_writing_builder.compile()

    return paper_writing_graph

# llm = ChatOpenAI(
#     openai_api_key=os.getenv("OPENAI_API_KEY"),
#     base_url="https://api.metisai.ir/openai/v1",
#     model="gpt-4.1-mini", 
#     temperature=0
# )

# print(llm.invoke("hi, how are you?"))

# for s in research_graph().stream(
#     {"messages": [("user", "when is Taylor Swift's next tour?")]},
#     {"recursion_limit": 100},
# ):
#     print(s)
#     print("---")

# for s in paper_writing_graph().stream(
#     {
#         "messages": [
#             (
#                 "user",
#                 "Write an outline for poem about cats and then write the poem to disk.",
#             )
#         ]
#     },
#     {"recursion_limit": 100},
# ):
#     print(s)
#     print("---")


