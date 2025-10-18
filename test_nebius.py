from langchain_nebius import ChatNebius
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv

load_dotenv()

nebius = ChatNebius(model="meta-llama/Meta-Llama-3.1-405B-Instruct")

@tool
def magic_function(a: int, b: int) -> int:
   """Magic function with unknown logic."""
   return a * b + 2

tools = [magic_function]

agent = create_react_agent(
    model=nebius,
    tools=tools,
    prompt="You are a helpful assistant.",
)

response = agent.invoke({"messages": [("user", "what is the magic function of 2 and 3?")]})

print(response["messages"][-1].content)