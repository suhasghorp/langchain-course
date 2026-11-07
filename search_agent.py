from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI
from tavily import TavilyClient
from langchain_tavily import TavilySearch

load_dotenv()

tavily = TavilyClient()


@tool()
def search(query: str) -> str:
    """
    Tool that searches over the internet
    Args:
        query: the query to search for
    Returns:
        The search result
    """

    print(f"Searching for {query}")
    # return "Tokyo weather is sunny"
    return tavily.search(query=query)


llm = AzureChatOpenAI(azure_deployment="gpt-4.1", api_version="2024-05-01-preview")
# tools = [search] -- needed only if you are not using langchain-tavily tool
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools)


def main():
    result = agent.invoke(
        {"messages": [HumanMessage(content="What is the current weather in Tokyo?")]}
    )
    print(result)


if __name__ == "__main__":
    main()
