from typing_extensions import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    get_buffer_string,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

from deep_research.prompt import (
    clarify_with_user_instructions,
    transform_messages_into_research_topic_prompt,
)
from deep_research.state import (
    AgentState,
    ClarifyWithUser,
    ResearchQuestion,
    AgentInputState,
)

from deep_research.utils import get_today_str

_DEFAULT_RESEARCH_MODEL = "anthropic:claude-haiku-4-5-20251001"


def clarifywithusers(
    state: AgentState,
    config: RunnableConfig,
) -> Command[Literal["write_research_brief", "__end__"]]:

    cfg = config.get("configurable", {})
    allow_clarification = cfg.get("allow_clarification", True)

    if not allow_clarification:
        return Command(
            goto="write_research_brief",
            update={
                "messages": [
                    AIMessage(content="Starting research now.")
                ]
            },
        )

    model_name = cfg.get(
        "research_model",
        _DEFAULT_RESEARCH_MODEL,
    )

    model = init_chat_model(
        model_name,
        temperature=0.7,
    )

    structured_output_model = model.with_structured_output(
        ClarifyWithUser
    )

    response = structured_output_model.invoke(
        [
            HumanMessage(
                content=clarify_with_user_instructions.format(
                    messages=get_buffer_string(
                        state["messages"]
                    ),
                    date=get_today_str(),
                )
            )
        ]
    )

    if response.need_clarification:
        return Command(
            goto=END,
            update={
                "messages": [
                    AIMessage(content=response.question)
                ]
            },
        )

    return Command(
        goto="write_research_brief",
        update={
            "messages": [
                AIMessage(content=response.verification)
            ]
        },
    )


def write_research_brief(
    state: AgentState,
    config: RunnableConfig,
):

    model_name = config.get(
        "configurable",
        {},
    ).get(
        "research_model",
        _DEFAULT_RESEARCH_MODEL,
    )

    model = init_chat_model(
        model_name,
        temperature=0.7,
    )

    structured_output_model = model.with_structured_output(
        ResearchQuestion
    )

    response = structured_output_model.invoke(
        [
            HumanMessage(
                content=transform_messages_into_research_topic_prompt.format(
                    messages=get_buffer_string(
                        state.get("messages", [])
                    ),
                    date=get_today_str(),
                )
            )
        ]
    )

    return {
        "research_brief": response.research_brief,
        "supervisor_messages": [
            HumanMessage(
                content=f"{response.research_brief}."
            )
        ],
    }


deep_researcher_builder = StateGraph(
    AgentState,
    input_schema=AgentInputState,
)

deep_researcher_builder.add_node(
    "clarify_with_user",
    clarifywithusers,
)

deep_researcher_builder.add_node(
    "write_research_brief",
    write_research_brief,
)

deep_researcher_builder.add_edge(
    START,
    "clarify_with_user",
)

deep_researcher_builder.add_edge(
    "write_research_brief",
    END,
)

scope_research = deep_researcher_builder.compile()