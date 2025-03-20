import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool
from langchain_openai import ChatOpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

search_tool = SerperDevTool()




def create_research_agent():
    
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    
    return Agent(
        role="Research Specialist",
        goal = "Conduct through research on give topics",
        backstory = "You are an experianced researcher with expertise in finding and synthesizing information from various websites",
        verbose=True,
        allow_delegation=False,
        tools = [search_tool],
        llm=llm
    )

def create_research_task(agent, topic):
    return Task(
        agent= agent,
        description = f"Research the following topic and provide a comprehensive summayr: {topic}",
        expected_output = "A detailed summary of the research findings, including key points and insights",
    )
    
def run_research(topic):
    agent = create_research_agent()
    task=create_research_task(agent, topic)
    crew = Crew(agents=[agent], tasks=[task])
    results = crew.kickoff()
    return results

if __name__ == "__main__":
    print("Welcome to the Research Agents")
    topic = input("Enter the research topic: ")
    result= run_research(topic)
    print("Research Results: ")
    print("Result: " + result)