from langchain_nebius import ChatNebius
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
from pubmed import PubMedAPI
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage, HumanMessage
from typing_extensions import TypedDict, Annotated
import operator
from typing import Literal
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display
from doi2pdf import PDFFromDOI
import pymupdf4llm
import mlflow
mlflow.langchain.autolog()

load_dotenv()

llm = ChatNebius(model="moonshotai/Kimi-K2-Instruct")
pubmed_api = PubMedAPI()

@tool
def search_pubmed(query: str, max_results: int = 5) -> list[dict]:
   """Search PubMed for meta-analyses on the given query."""
   print(f"Searching PubMed for {query} with max_results={max_results}")
   papers = pubmed_api.search(query, max_results=max_results, meta_analysis_only=True)
   print(f"Found {len(papers)} papers")
   return papers

@tool
def submit_doi(doi: str, goal: str) -> str:
    """Call this when you have found a paper that fits the goal. (if you get an error, you must find another paper.)"""
    print(f"Paper DOI found: https://doi.org/{doi}")
    pdf_from_doi = PDFFromDOI()
    try:
        path = pdf_from_doi.download(doi)
    except Exception as e:
        error = f"Error downloading paper, please try to find another one that fits the goal.\nError: {e}"
        print(error)
        return error

    try:
        md = pymupdf4llm.to_markdown(str(path))
    except Exception as e:
        error = f"Error processing the doi, please try to find another one that fits the goal.\nError: {e}"
        print(error)
        return error

    feedback = llm.invoke([SystemMessage(content=f"You are a helpful, intelligent assistant. You are tasked to evaluate if the following paper is a good fit for the goal: {goal}.\n\n Reply with 'yes' if it fits; otherwise, don't just answer no, but explain why not. If parts of the paper are missing, reply: 'There was an error processing the paper, please try to find another one that fits the goal.'"), HumanMessage(content=f"This is the paper:\n\n {md}")])

    feedback_text = feedback.content.lower().strip()
    print(feedback_text)
    if feedback_text in ("y", "yes"):
        return "success"
    else:
        return feedback.content

tools = [search_pubmed, submit_doi]
tools_by_name = {tool.name: tool for tool in tools}
llm_with_tools = llm.bind_tools(tools)


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    goal: str

def llm_call(state: dict):
    """LLM decides whether to call a tool or not"""
    return {
        "messages": [
            llm_with_tools.invoke(
                [SystemMessage(content="You must call submit_doi with the DOI when you find a suitable paper. (answering the doi in the chat is not enough! The tool must be called!)")]
                + state["messages"]
            )
        ]
    }

def tool_node(state: dict):
    """Performs the tool call"""
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        args = tool_call["args"]
        if tool_call["name"] == "submit_doi":
            args["goal"] = state["goal"]
        observation = tool.invoke(args)
        result.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))
    return {"messages": result}


def should_continue(state: MessagesState) -> Literal["tool_node", "llm_call"]:
    """Route to tool_node if LLM made tool calls, else loop back"""
    last_message = state["messages"][-1]
    return "tool_node" if last_message.tool_calls else "llm_call"

def check_finish(state: MessagesState) -> Literal["llm_call", END]:
    """Check if submit_doi returned success, if so end, else continue"""
    for msg in reversed(state["messages"]):
        if isinstance(msg, ToolMessage) and msg.content == "success":
            return END
    return "llm_call"

# Build workflow
agent_builder = StateGraph(MessagesState)

# Add nodes
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

# Add edges to connect nodes
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges("llm_call", should_continue)
agent_builder.add_conditional_edges("tool_node", check_finish)

# Compile the agent
agent = agent_builder.compile()


if __name__ == "__main__":
    goal = "Find a high quality meta-analysis paper that measures the effect of age at menopause on CVD"
    messages = agent.invoke({"messages": [HumanMessage(content=goal)], "goal": goal})
    for m in messages["messages"]:
        if not isinstance(m, ToolMessage):
            m.pretty_print()