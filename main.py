from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from schemas import AgentResponse


def main():
    print("Hello from langchain-course!")

    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # Tavily tool (new, recommended package)
    tools = [TavilySearch(max_results=10)]

    # Structured output parser
    parser = PydanticOutputParser(pydantic_object=AgentResponse)

    # IMPORTANT: do NOT f-string the format instructions directly into the template.
    # Use a placeholder + partial() so LangChain won't treat $defs/properties as variables.
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a job search assistant.\n"
            "Use the search tool to find real job postings.\n"
            "LinkedIn may be login-blocked; if so, use other public sources.\n"
            "Return EXACTLY 3 job postings with details + include source URLs.\n\n"
            "{format_instructions}"
        ),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]).partial(format_instructions=parser.get_format_instructions())

    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    question = (
        "Search for 3 job postings for an AI Data Scientist in Banglore on LinkedIn "
        "and list their details (title, company, location, link, posted date if available)."
    )

    result = agent_executor.invoke({"input": question})
    text_output = result.get("output", "")

    # Parse into AgentResponse
    try:
        structured = parser.parse(text_output)
        print(structured.model_dump_json(indent=2))
    except Exception as e:
        print("Could not parse structured output. Printing raw output instead.")
        print("Parser error:", e)
        print(text_output)


if __name__ == "__main__":
    main()
