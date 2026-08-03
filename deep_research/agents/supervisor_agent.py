import asyncio

from typing_extensions import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    HumanMessage,
    BaseMessage,
    SystemMessage,
    ToolMessage,
    filter_messages,
)

from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

from deep_research.prompt import lead_researcher_prompt
from deep_research.agents.research_agent import researcher_agent
from deep_research.state import (
    SupervisorState,
    ConductResearch,
    ResearchComplete,
)
from deep_research.utils import (
    get_today_str,
    think_tool,
)

try:
    import nest_asyncio

    try:
        from IPython import get_ipython

        if get_ipython() is not None:
            nest_asyncio.apply()

    except ImportError:
        pass

except ImportError:
    pass


_DEFAULT_RESEARCH_MODEL = "anthropic:claude-haiku-4-5-20251001"

_SUPERVISOR_TOOLS = [
    ConductResearch,
    ResearchComplete,
    think_tool,
]

max_researcher_iterations = 6
max_concurrent_researchers = 3


def get_notes_from_tool_calls(
    messages: list[BaseMessage],
) -> list[str]:
    return [
        tool_msg.content
        for tool_msg in filter_messages(
            messages,
            include_types="tool",
        )
    ]


async def supervisor(
    state: SupervisorState,
    config: RunnableConfig,
) -> Command[Literal["supervisor_tools"]]:

    supervisor_messages = state.get(
        "supervisor_messages",
        [],
    )

    cfg = config.get("configurable", {})

    model_name = cfg.get(
        "research_model",
        _DEFAULT_RESEARCH_MODEL,
    )

    concurrent = cfg.get(
        "max_concurrent_research_units",
        max_concurrent_researchers,
    )

    iterations = cfg.get(
        "max_researcher_iterations",
        max_researcher_iterations,
    )

    supervisor_model = init_chat_model(
        model_name,
        temperature=0.0,
    )

    supervisor_model_with_tools = supervisor_model.bind_tools(
        _SUPERVISOR_TOOLS
    )

    system_message = lead_researcher_prompt.format(
        date=get_today_str(),
        max_concurrent_research_units=concurrent,
        max_researcher_iterations=iterations,
    )

    messages = [
        SystemMessage(content=system_message)
    ] + supervisor_messages

    response = await supervisor_model_with_tools.ainvoke(
        messages
    )

    return Command(
        goto="supervisor_tools",
        update={
            "supervisor_messages": [response],
            "research_iterations": state.get(
                "research_iterations",
                0,
            )
            + 1,
        },
    )
    
async def supervisor_tools(
    state: SupervisorState,
    config: RunnableConfig,
) -> Command[Literal["supervisor", "__end__"]]:

    supervisor_messages = state.get(
        "supervisor_messages",
        []
    )

    research_iterations = state.get(
        "research_iterations",
        0
    )

    most_recent_message = supervisor_messages[-1]

    cfg = config.get("configurable", {})

    iterations_limit = cfg.get(
        "max_researcher_iterations",
        max_researcher_iterations,
    )

    tool_messages = []
    all_raw_notes = []

    next_step = "supervisor"
    should_end = False

    exceeded_iterations = (
        research_iterations >= iterations_limit
    )

    no_tool_calls = not most_recent_message.tool_calls

    research_complete = any(
        tool_call["name"] == "ResearchComplete"
        for tool_call in most_recent_message.tool_calls
    )

    if (
        exceeded_iterations
        or no_tool_calls
        or research_complete
    ):
        should_end = True
        next_step = END

    else:
        try:

            think_tool_calls = [
                tool_call
                for tool_call in most_recent_message.tool_calls
                if tool_call["name"] == "think_tool"
            ]

            conduct_research_calls = [
                tool_call
                for tool_call in most_recent_message.tool_calls
                if tool_call["name"] == "ConductResearch"
            ]

            for tool_call in think_tool_calls:

                observation = think_tool.invoke(
                    tool_call["args"]
                )

                tool_messages.append(
                    ToolMessage(
                        content=observation,
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"],
                    )
                )

            if conduct_research_calls:

                coros = [
                    researcher_agent.ainvoke(
                        {
                            "researcher_messages": [
                                HumanMessage(
                                    content=tool_call["args"]["research_topic"]
                                )
                            ],
                            "research_topic": tool_call["args"]["research_topic"],
                        },
                        config=config,
                    )
                    for tool_call in conduct_research_calls
                ]

                tool_results = await asyncio.gather(
                    *coros
                )

                research_tool_messages = [
                    ToolMessage(
                        content=result.get(
                            "compressed_research",
                            "Error synthesizing research report",
                        ),
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"],
                    )
                    for result, tool_call in zip(
                        tool_results,
                        conduct_research_calls,
                    )
                ]

                tool_messages.extend(
                    research_tool_messages
                )

                all_raw_notes = [
                    "\n".join(
                        result.get(
                            "raw_notes",
                            [],
                        )
                    )
                    for result in tool_results
                ]

        except Exception as e:
            print(
                f"Error in supervisor tools: {e}"
            )

            should_end = True
            next_step = END

    if should_end:

        return Command(
            goto=next_step,
            update={
                "notes": get_notes_from_tool_calls(
                    supervisor_messages
                ),
                "research_brief": state.get(
                    "research_brief",
                    "",
                ),
            },
        )

    return Command(
        goto=next_step,
        update={
            "supervisor_messages": tool_messages,
            "raw_notes": all_raw_notes,
        },
    )


supervisor_builder = StateGraph(
    SupervisorState,
)

supervisor_builder.add_node(
    "supervisor",
    supervisor,
)

supervisor_builder.add_node(
    "supervisor_tools",
    supervisor_tools,
)

supervisor_builder.add_edge(
    START,
    "supervisor",
)

supervisor_agent = supervisor_builder.compile()