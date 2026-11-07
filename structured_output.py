from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI
from tavily import TavilyClient
from langchain_tavily import TavilySearch
from typing import List
from pydantic import BaseModel, Field

load_dotenv()

tavily = TavilyClient()

class Source(BaseModel):
    """Schema for a source used by the agent"""
    url:str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""
    answer:str = Field(description="The agent's answer to the query")
    sources:List[Source] = Field(default_factory=list, description="List of sources used to generate the answer")

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
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)


def main():
    '''result = agent.invoke(
        {"messages": [HumanMessage(content="What is the current weather in Tokyo?")]}
    )'''
    result = agent.invoke(
        {"messages": [HumanMessage(content="search for 3 job postings for an AI engineer in the bay area on LinkedIn")]}
    )
    print(result)
    print(result['structured_response'].sources)


if __name__ == "__main__":
    main()
