from typing import Annotated
import os

from langchain_core.tools import tool, InjectedToolCallId
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import create_react_agent, InjectedState
from langgraph.types import Command, Send

from src.MAS_supervisor.agents.math_agent import math_agent
from src.MAS_supervisor.agents.research_agent import research_agent
from src.MAS_supervisor.utils.pretty_print import pretty_print_message, pretty_print_messages

class MAS_Supervisor:
    def __init__(self, api_key):
        self.api_key = api_key
        self.llm = ChatOpenAI(
           openai_api_key=os.getenv("OPENAI_API_KEY"),
           base_url="https://api.metisai.ir/openai/v1",
           model="gpt-4.1-mini", 
           temperature=0
        )
        self.research_agent = research_agent(self.llm)
        self.math_agent = math_agent(self.llm)

        self.assign_to_research_agent = self._create_handoff_tool(
            agent_name="research_agent",
            description="Assign task to a researcher agent."
        )

        self.assign_to_math_agent = self._create_handoff_tool(
            agent_name="math_agent",
            description="Assign task to a math agent."
        )

        self.supervisor_agent = self.create_supervisor()
        self.MAS_graph = self.create_multi_agent_graph()
        
    def create_supervisor(self):
        return create_react_agent(
            model=self.llm,
            tools=[self.assign_to_research_agent, self.assign_to_math_agent],
            prompt=(
                "You are a supervisor managing two agents:\n"
                "- a research agent. Assign research-related tasks to this agent\n"
                "- a math agent. Assign math-related tasks to this agent\n"
                "Assign work to one agent at a time, do not call agents in parallel.\n"
                "Do not do any work yourself."

            ),
            name="Supervisor",
        )

    def create_multi_agent_graph(self):
        supervisor = (
            StateGraph(MessagesState)
            # NOTE: `destinations` is only needed for visualization and doesn't affect runtime behavior
            .add_node(self.supervisor_agent, destinations=("research_agent", "math_agent"))
            .add_node(self.research_agent)
            .add_node(self.math_agent)
            .add_edge(START, "Supervisor")
            # always return back to the supervisor
            .add_edge("research_agent", "Supervisor")
            .add_edge("math_agent", "Supervisor")
            .compile()
        )
        return supervisor

    @staticmethod
    def _create_handoff_tool(*, agent_name: str, description: str | None = None):
        name = f"transfer_to_{agent_name}"
        description = description or f"Ask {agent_name} for help."

        @tool(name, description=description)
        def handoff_tool(
            state: Annotated[MessagesState, InjectedState],
            tool_call_id: Annotated[str, InjectedToolCallId],
            ) -> Command:
            tool_message = {
                "role": "tool",
                "name": name,
                "content": f"Successfully transferred to {agent_name}.",
                "tool_call_id": tool_call_id,
            }
            # highlight next line
            return Command(
                # highlight next line
                goto=agent_name, # (1)!
                # highlight next line
                update={**state, "messages": state["messages"] + [tool_message]}, # (2)!
                # highlight next line
                graph=Command.PARENT, #(3)!
            )
    
        return handoff_tool
    
    @staticmethod
    def create_task_description_handoff_tool(
    *, agent_name: str, description: str | None = None
    ):
        name = f"transfer_to_{agent_name}"
        description = description or f"Ask {agent_name} for help."

        @tool(name, description=description)
        def handoff_tool(
            # this is populated by the supervisor LLM
            task_description: Annotated[
                str,
                "Description of what the next agent should do, including all of the relevant context.",
            ],
            # these parameters are ignored by the LLM
            state: Annotated[MessagesState, InjectedState],
        ) -> Command:
            task_description_message = {"role": "user", "content": task_description}
            agent_input = {**state, "messages": [task_description_message]}
            return Command(
                # highlight-next-line
                goto=[Send(agent_name, agent_input)],
                graph=Command.PARENT,
            )

        return handoff_tool

    def call_multi_agent_graph(self, input):
        for chunk in self.MAS_graph.stream(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": input,
                    }
                ]
            },
        ):
            pretty_print_messages(chunk, last_message=True)

        # print("\n\n=== Final Message History ===\n")
        # final_message_history = chunk["Supervisor"]["messages"]
        # for message in final_message_history:
        #     message.pretty_print()

# MAS_Supervisor = MAS_Supervisor(api_key=os.getenv("OPENAI_API_KEY"))
# MAS_Supervisor.call_multi_agent_graph("find US and New York state GDP in 2024. what % of US GDP was New York state?")
