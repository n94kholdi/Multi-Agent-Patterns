import os
from langchain_core.messages import BaseMessage
from src.MAS_hierarichal.core.teams_graph import *


class orchestrate_graph:
    def __init__(self, api_key):
        self.api_key = api_key
        os.environ["OPENAI_API_KEY"] = api_key 

        self.llm = ChatOpenAI(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            base_url="https://api.metisai.ir/openai/v1",
            model="gpt-4.1-mini", 
            temperature=0
        )

        self.teams_supervisor_node = make_supervisor_node(llm, ["research_team", "writing_team"])

    def call_research_team(self, state: State) -> Command[Literal["supervisor"]]:
        response = research_graph().invoke({"messages": state["messages"][-1]})
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


    def call_paper_writing_team(self, state: State) -> Command[Literal["supervisor"]]:
        response = paper_writing_graph().invoke({"messages": state["messages"][-1]})
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


    def create_graph(self,):
        # Define the graph.
        super_builder = StateGraph(State)
        super_builder.add_node("supervisor", self.teams_supervisor_node)
        super_builder.add_node("research_team", self.call_research_team)
        super_builder.add_node("writing_team", self.call_paper_writing_team)

        super_builder.add_edge(START, "supervisor")
        super_graph = super_builder.compile()

        return super_graph
    
    def process_input(self, prompt):

        responses = []
        for stream_ in self.create_graph().stream(
            {
                "messages":[
                    ("user", prompt)
                ]
            },
            {"recursion_limit": 150},
        ):
            print(stream_)
            print("---")
            responses.append(stream_)        
        return responses

# orchestrate_graph().process_input("Research AI agents and write a brief report about them.")
# for s in orchestrate_graph().create_graph().stream(
#     {
#         "messages": [
#             ("user", "Research AI agents and write a brief report about them.")
#         ],
#     },
#     {"recursion_limit": 150},
# ):
#     print(s)
#     print("---")