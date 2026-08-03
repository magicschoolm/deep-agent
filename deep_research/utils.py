from datetime import datetime
from typing import Annotated as Ann

from typing_extensions import Annotated, List, Literal

from tavily import TavilyClient

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool, InjectedToolArg
from deep_research.state import Summary
#from deep_research.state import Summary
from deep_research.prompt import summarize_webpage_prompt


_DEFAULT_SUMMARIZATION_MODEL = "anthropic:claude-haiku-4-5-20251001"


def get_today_str() -> str:
    now = datetime.now()
    return f"{now:%a, %b %d, %Y}"


tavily_client = TavilyClient()


def tavily_search_multiple(
    search_queries: List[str],
    max_results: int = 3,
    topic: Literal["general", "news"] = "general",
    include_raw_content: bool = True,
) -> List[dict]:

    search_docs = []

    for query in search_queries:

        result = tavily_client.search(
            query=query,
            max_results=max_results,
            include_raw_content=include_raw_content,
            topic=topic,
        )

        search_docs.append(result)

    return search_docs


def summarize_webpage_content(
    webpage_content: str,
    model_name: str = _DEFAULT_SUMMARIZATION_MODEL,
) -> str:

    try:

        summarization_model = init_chat_model(
            model_name,
            temperature=0.0,
        )

        structured_model = summarization_model.with_structured_output(
            Summary
        )

        summary = structured_model.invoke(
            [
                HumanMessage(
                    content=summarize_webpage_prompt.format(
                        webpage_content=webpage_content,
                        date=get_today_str(),
                    )
                )
            ]
        )

        formatted_summary = (
            f"<summary>\n{summary.summary}\n</summary>\n\n"
            f"<key_excerpts>\n{summary.key_excerpts}\n</key_excerpts>"
        )

        return formatted_summary

    except Exception as e:

        print(f"Failed to summarize webpage: {e}")

    if len(webpage_content) > 1000:
        return webpage_content[:1000] + "..."

    return webpage_content


def deduplicate_search_results(
    search_results: List[dict],
) -> dict:

    unique_results = {}

    for response in search_results:

        for result in response["results"]:

            url = result["url"]

            if url not in unique_results:
                unique_results[url] = result

    return unique_results


def process_search_results(
    unique_results: dict,
) -> dict:

    summarized_results = {}

    for url, result in unique_results.items():

        if not result.get("raw_content"):
            content = result["content"]

        else:
            content = summarize_webpage_content(
                result["raw_content"]
            )

        summarized_results[url] = {
            "title": result["title"],
            "content": content,
        }

    return summarized_results

def format_search_output(
    summarized_results: dict,
) -> str:

    if not summarized_results:
        return "No search results found."

    formatted_output = "Search Results:\n\n"

    for i, (url, result) in enumerate(
        summarized_results.items(),
        start=1,
    ):
        formatted_output += (
            f"--- SOURCE {i}: {result['title']} ---\n"
        )
        formatted_output += f"URL: {url}\n\n"
        formatted_output += (
            f"SUMMARY:\n{result['content']}\n\n"
        )
        formatted_output += "-" * 80 + "\n\n"

    return formatted_output


@tool(parse_docstring=False)
def Tavily_search(
    query: str,
    config: RunnableConfig,
    max_results: Ann[int, InjectedToolArg] = 3,
    topic: Ann[
        Literal["general", "news"],
        InjectedToolArg,
    ] = "general",
) -> str:
    """
    Search the web using Tavily.
    """

    summarization_model = config.get(
        "configurable",
        {},
    ).get(
        "summarization_model",
        _DEFAULT_SUMMARIZATION_MODEL,
    )

    search_results = tavily_search_multiple(
        [query],
        max_results=max_results,
        topic=topic,
        include_raw_content=True,
    )

    unique_results = deduplicate_search_results(
        search_results
    )

    summarized_results = {}

    for url, result in unique_results.items():

        if not result.get("raw_content"):
            content = result["content"]

        else:
            content = summarize_webpage_content(
                result["raw_content"],
                summarization_model,
            )

        summarized_results[url] = {
            "title": result["title"],
            "content": content,
        }

    return format_search_output(
        summarized_results
    )


@tool(parse_docstring=False)
def think_tool(
    reflection: str,
) -> str:
    """Internal reasoning tool."""

    return (
        f"Reflection recorded: {reflection}"
    )