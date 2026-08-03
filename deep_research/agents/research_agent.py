from typing_extensions import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    ToolMessage,
    filter_messages,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END

from deep_research.state import (
    ResearcherState,
    ResearcherOutputState,
)
from deep_research.utils import (
    Tavily_search,
    get_today_str,
    think_tool,
)
from deep_research.prompt import (
    research_system_prompt,
    compress_research_system_prompt,
    compress_research_simple_human_message,
)

_DEFAULT_RESEARCH_MODEL = "anthropic:claude-haiku-4-5-20251001"
_DEFAULT_COMPRESSION_MODEL = "anthropic:claude-haiku-4-5-20251001"

tools = [Tavily_search, think_tool]


def llm_call(state: ResearcherState, config: RunnableConfig):
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

    model_with_tools = model.bind_tools(tools)

    return {
        "researcher_messages": [
            model_with_tools.invoke(
                [SystemMessage(content=research_system_prompt)]
                + state["researcher_messages"]
            )
        ]
    }


def tool_node(state: ResearcherState, config: RunnableConfig):
    tool_calls = state["researcher_messages"][-1].tool_calls

    observations = [
        Tavily_search.invoke(
            tool_call["args"],
            config=config,
        )
        for tool_call in tool_calls
    ]

    tool_output = [
        ToolMessage(
            content=observation,
            name=tool_call["name"],
            tool_call_id=tool_call["id"],
        )
        for observation, tool_call in zip(
            observations,
            tool_calls,
        )
    ]

    i = state.get("tool_call_i", 0) + 1

    return {
        "researcher_messages": tool_output,
        "tool_call_i": i,
    }


def compress_research(
    state: ResearcherState,
    config: RunnableConfig,
) -> dict:

    model_name = config.get(
        "configurable",
        {},
    ).get(
        "compression_model",
        _DEFAULT_COMPRESSION_MODEL,
    )

    compress_model = init_chat_model(
        model_name,
        temperature=0.7,
    )

    system_message = compress_research_system_prompt.format(
        date=get_today_str()
    )

    messages = (
        [SystemMessage(content=system_message)]
        + state.get("researcher_messages", [])
        + [
            HumanMessage(
                content=compress_research_simple_human_message
            )
        ]
    )

    response = compress_model.invoke(messages)

    raw_notes = [
        str(message.content)
        for message in filter_messages(
            state["researcher_messages"],
            include_types=["tool", "ai"],
        )
    ]

    return {
        "compressed_research": str(response.content),
        "raw_notes": [
            "\n".join(raw_notes)
        ],
    }


def should_continue(
    state: ResearcherState,
    config: RunnableConfig,
) -> Literal["tool_node", "compress_research"]:

    last_message = state["researcher_messages"][-1]

    max_calls = config.get(
        "configurable",
        {},
    ).get(
        "max_react_tool_calls",
        5,
    )

    iterations = state.get(
        "tool_call_i",
        0,
    )

    if last_message.tool_calls and iterations < max_calls:
        return "tool_node"

    return "compress_research"


agent_builder = StateGraph(
    ResearcherState,
    output_schema=ResearcherOutputState,
)

agent_builder.add_node(
    "llm_call",
    llm_call,
)

agent_builder.add_node(
    "tool_node",
    tool_node,
)

agent_builder.add_node(
    "compress_research",
    compress_research,
)

agent_builder.add_edge(
    START,
    "llm_call",
)

agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    {
        "tool_node": "tool_node",
        "compress_research": "compress_research",
    },
)

agent_builder.add_edge(
    "tool_node",
    "llm_call",
)

agent_builder.add_edge(
    "compress_research",
    END,
)

researcher_agent = agent_builder.compile()