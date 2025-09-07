import os
from langchain_core.messages import BaseMessage
from src.MAS_hierarichal.core.teams_graph import *


def orchestrate_graph():

    os.environ["OPENAI_API_KEY"] = "tpsg-QS1lyFgvVj8qu21PlxfFaZLtAtNxpY3"

    llm = ChatOpenAI(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        base_url="https://api.metisai.ir/openai/v1",
        model="gpt-4.1-mini", 
        temperature=0
    )

    teams_supervisor_node = make_supervisor_node(llm, ["research_team", "writing_team"])
    def call_research_team(state: State) -> Command[Literal["supervisor"]]:
        response = research_graph.invoke({"messages": state["messages"][-1]})
        return Command(
            update={
                "messages": [
                    HumanMessage(
                        content=response["messages"][-1].content, name="research_team"
                    )
                ]
            },
            goto="supervisor",
        )


    def call_paper_writing_team(state: State) -> Command[Literal["supervisor"]]:
        response = paper_writing_graph.invoke({"messages": state["messages"][-1]})
        return Command(
            update={
                "messages": [
                    HumanMessage(
                        content=response["messages"][-1].content, name="writing_team"
                    )
                ]
            },
            goto="supervisor",
        )


    # Define the graph.
    super_builder = StateGraph(State)
    super_builder.add_node("supervisor", teams_supervisor_node)
    super_builder.add_node("research_team", call_research_team)
    super_builder.add_node("writing_team", call_paper_writing_team)

    super_builder.add_edge(START, "supervisor")
    super_graph = super_builder.compile()

    return super_graph