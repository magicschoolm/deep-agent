import operator
from typing import Optional, Sequence

from typing_extensions import Annotated, List, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages

from pydantic import BaseModel, Field


class AgentInputState(MessagesState):
    pass


class AgentState(MessagesState):
    research_brief: Optional[str]
    supervisor_messages: Annotated[Sequence[BaseMessage], add_messages]
    raw_notes: Annotated[List[str], operator.add] = []
    notes: Annotated[List[str], operator.add] = []
    final_report: str


class ResearcherState(TypedDict):
    researcher_messages: Annotated[Sequence[BaseMessage], add_messages]
    tool_call_i: int
    research_topic: str
    compressed_research: str
    raw_notes: Annotated[List[str], operator.add]


class ResearcherOutputState(TypedDict):
    compressed_research: str
    raw_notes: Annotated[List[str], operator.add]
    researcher_messages: Annotated[Sequence[BaseMessage], add_messages]


class SupervisorState(TypedDict):
    supervisor_messages: Annotated[Sequence[BaseMessage], add_messages]
    research_brief: str
    notes: Annotated[List[str], operator.add]
    research_iterations: int
    raw_notes: Annotated[List[List[str]], operator.add]


class ClarifyWithUser(BaseModel):
    need_clarification: bool = Field(
        description="Whether the user needs clarification."
    )
    question: str = Field(
        description="Clarification question for the user."
    )
    verification: str = Field(
        description="Confirmation message when clarification is not needed."
    )


class ResearchQuestion(BaseModel):
    research_brief: str = Field(
        description="Research brief generated from the conversation."
    )


class Summary(BaseModel):
    summary: str = Field(
        description="Concise summary of the webpage."
    )
    key_excerpts: str = Field(
        description="Important excerpts from the webpage."
    )


class ConductResearch(BaseModel):
    research_topic: str = Field(
        description="Detailed topic to research."
    )


class ResearchComplete(BaseModel):
    pass