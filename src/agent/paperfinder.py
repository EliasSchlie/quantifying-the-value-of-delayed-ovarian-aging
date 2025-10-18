from langchain_nebius import ChatNebius
from dotenv import load_dotenv
from pubmed import PubMedAPI
from langchain_core.messages import SystemMessage, HumanMessage
from typing_extensions import TypedDict, Annotated
from typing import Literal
from langgraph.graph import StateGraph, START, END
from doi2pdf import PDFFromDOI
import pymupdf4llm
import mlflow
from interaction_storage import InteractionStorage

mlflow.langchain.autolog()
load_dotenv()

llm = ChatNebius(model="moonshotai/Kimi-K2-Instruct")
pubmed_api = PubMedAPI()
pdf_from_doi = PDFFromDOI()
interaction_storage = InteractionStorage()

# State definition
class GraphState(TypedDict):
    disease_of_interest: str
    query: str
    papers: list[dict]
    checked_dois: Annotated[list[str], lambda x, y: list(set(x + y))]
    tried_queries: Annotated[list[str], lambda x, y: x + y]
    current_paper: dict
    paper_md: str
    metrics_count: int
    min_metrics: int

# Node functions
def create_query(state: GraphState) -> dict:
    """AI creates PubMed query for menopause timing and health outcomes"""
    print(f"\n--- Creating query for: {state['disease_of_interest']} ---")
    # return {"query": "10.1016/S2468-2667(19)30155-0"}
    tried = state.get("tried_queries", [])
    
    if tried:
        print(f"Previous queries tried: {len(tried)}")
        previous_queries_text = "\n".join([f"  {i+1}. {q}" for i, q in enumerate(tried)])
        prompt = f"""Disease/outcome: {state['disease_of_interest']}

Previously tried queries:
{previous_queries_text}

Create a NEW query focusing on menopause timing (age at menopause, early menopause, premature menopause) and its association with this health outcome.

Be creative with terminology:
- Menopause timing variations (early menopause, premature menopause, age at natural menopause, ANM)
- Include meta-analyses, cohort studies, case-control studies
- Use related terms for the health outcome
- Consider cardiovascular, bone health, cognitive outcomes

Create a concise PubMed search query."""
    else:
        prompt = f"""Disease/outcome: {state['disease_of_interest']}

Create a PubMed search query to find studies linking menopause timing (age at menopause, early menopause) to this health outcome.
Focus on studies with risk metrics (OR, HR, RR), meta-analyses, and cohort/case-control studies. Include relevant keywords and filters."""
    
    response = llm.invoke([
        SystemMessage(content="You are an expert at crafting PubMed queries. Find studies linking menopause timing to specific health outcomes with quantitative risk metrics."),
        HumanMessage(content=prompt)
    ])
    
    query = response.content.strip()
    print(f"Generated query: {query}")
    
    return {"query": query, "tried_queries": [query]}

def search_pubmed(state: GraphState) -> dict:
    """Search PubMed API"""
    print(f"\n--- Searching PubMed: {state['query']} ---")
    papers = pubmed_api.search(state['query'], max_results=100)
    print(f"Found {len(papers)} papers")
    return {"papers": papers}

def filter_papers(state: GraphState) -> dict:
    """Filter out already checked papers"""
    print("\n--- Filtering papers ---")
    checked = state.get("checked_dois", [])
    filtered = [p for p in state["papers"] if p.get("doi") and p["doi"] not in checked]
    print(f"Filtered to {len(filtered)} new papers (from {len(state['papers'])})")
    return {"papers": filtered}

def check_abstract(state: GraphState) -> dict:
    """AI checks if abstract is relevant"""
    if not state["papers"]:
        return {"current_paper": {}}
    
    paper = state["papers"][0]
    remaining = state["papers"][1:]
    
    print(f"\n--- Checking abstract: {paper.get('title', 'No title')[:50]}... ---")
    
    response = llm.invoke([
        SystemMessage(content=f"You are a paper classifier, that estimates a paper's relevance from it's abstract alone. \nEstimate if this paper studies menopause timing (age at menopause, early/premature menopause) and its association with {state['disease_of_interest']}. Relevant papers likely report quantitative risk metrics (OR, HR, RR) with confidence intervals. Reply 'yes' if relevant, 'no' if not."),
        HumanMessage(content=f"Title: {paper.get('title', '')}\n\nAbstract: {paper.get('abstract', '')}")
    ])
    
    is_relevant = response.content.strip().lower() in ["yes", "y"]
    print(f"Abstract relevant: {is_relevant}")
    
    if is_relevant:
        return {"papers": remaining, "current_paper": paper, "checked_dois": [paper.get("doi", "")]}
    else:
        return {"papers": remaining, "current_paper": {}, "checked_dois": [paper.get("doi", "")]}

def download_paper(state: GraphState) -> dict:
    """Download paper PDF and convert to markdown"""
    paper = state["current_paper"]
    doi = paper.get("doi")
    
    if not doi:
        return {"paper_md": "", "current_paper": {}}
    
    print(f"\n--- Downloading DOI: {doi} ---")
    
    try:
        path = pdf_from_doi.download(doi)
        md = pymupdf4llm.to_markdown(str(path))
        print(f"Successfully converted to markdown ({len(md)} chars)")
        return {"paper_md": md}
    except Exception as e:
        print(f"Error processing paper: {e}")
        return {"paper_md": "", "current_paper": {}}

def extract_interactions(state: GraphState) -> dict:
    """AI extracts risk metrics from paper using tool calls, looping until done"""
    if not state["paper_md"]:
        return {"metrics_count": state.get("metrics_count", 0), "current_paper": {}, "paper_md": ""}
    
    print("\n--- Extracting risk metrics ---")
    
    doi = state['current_paper'].get('doi', '')
    pub_date = state['current_paper'].get('pub_date', '')
    
    extraction_complete = False
    
    from langchain_core.tools import tool
    
    @tool
    def submit_risk_metric(menopause_timing_definition: str, health_outcome: str, metric_type: str, metric_value: str, ci_95: str) -> str:
        """Submit an extracted risk metric from the paper.
        
        Args:
            menopause_timing_definition: How menopause groups were defined (e.g., "Early < 45 years vs Normal 50-54 years" or "per 1-year decrease in age at menopause")
            health_outcome: Specific health outcome (e.g., "Ischemic Stroke", "Type 2 Diabetes", "Coronary Heart Disease")
            metric_type: Type of metric - must be "OR", "HR", or "RR"
            metric_value: The numerical value of the metric (e.g., "1.45", "2.1")
            ci_95: 95% confidence interval (e.g., "1.20-1.75", "1.8-2.4")
        """
        interaction_storage.add_risk_metric(menopause_timing_definition, health_outcome, metric_type, metric_value, ci_95, doi, pub_date)
        print(f"  ✓ Stored: {health_outcome} | {metric_type}={metric_value} (CI: {ci_95})")
        return "Risk metric submitted successfully. Continue extracting or call finish_extraction when done."
    
    @tool
    def finish_extraction() -> str:
        """Call this when you have finished extracting ALL relevant risk metrics from the paper, or if there are no relevant metrics to extract."""
        nonlocal extraction_complete
        extraction_complete = True
        return "Extraction complete."
    
    llm_with_tools = llm.bind_tools([submit_risk_metric, finish_extraction])
    
    initial_prompt = f"""Analyze this paper and extract ALL risk metrics (OR, HR, RR) linking menopause timing to health outcomes.

Target outcome: {state['disease_of_interest']}

Extract for EACH reported metric:
1. Menopause Timing Definition: How groups were defined (e.g., "Early < 45 years vs Normal 50-54 years", "per 1-year decrease in ANM")
2. Specific Health Outcome: Be precise (e.g., "Ischemic Stroke" not just "Stroke")
3. Metric Type: OR, HR, or RR
4. Metric Value: The numerical value
5. 95% CI: REQUIRED - the confidence interval

CRITICAL REQUIREMENTS:
- MUST have 95% CI reported (skip metrics without CI)
- Call submit_risk_metric for EACH metric
- Extract from main results AND subgroup analyses if present
- When done, call finish_extraction

Paper:
{state['paper_md']}"""
    
    messages = [
        SystemMessage(content="You are a scientific paper analyzer. Extract ALL risk metrics (OR, HR, RR) with their 95% CI by calling submit_risk_metric for each. Call finish_extraction when done."),
        HumanMessage(content=initial_prompt)
    ]
    
    count = state.get("metrics_count", 0)
    max_iterations = 20
    iteration = 0
    
    while not extraction_complete and iteration < max_iterations:
        iteration += 1
        print(f"\n  Extraction iteration {iteration}...")
        
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        
        if hasattr(response, 'tool_calls') and response.tool_calls:
            print(f"  {len(response.tool_calls)} tool call(s)")
            
            tool_messages = []
            for tool_call in response.tool_calls:
                tool_name = tool_call['name']
                
                if tool_name == 'submit_risk_metric':
                    try:
                        result = submit_risk_metric.invoke(tool_call['args'])
                        count += 1
                        tool_messages.append({
                            "role": "tool",
                            "content": result,
                            "tool_call_id": tool_call['id']
                        })
                    except Exception as e:
                        print(f"  ✗ Failed to submit metric: {e}")
                        tool_messages.append({
                            "role": "tool",
                            "content": f"Error: {e}",
                            "tool_call_id": tool_call['id']
                        })
                
                elif tool_name == 'finish_extraction':
                    result = finish_extraction.invoke({})
                    print(f"  ✓ {result}")
                    tool_messages.append({
                        "role": "tool",
                        "content": result,
                        "tool_call_id": tool_call['id']
                    })
            
            from langchain_core.messages import ToolMessage
            for tm in tool_messages:
                messages.append(ToolMessage(content=tm["content"], tool_call_id=tm["tool_call_id"]))
        else:
            print("  No tool calls, prompting to continue or finish...")
            messages.append(HumanMessage(content="Continue extracting risk metrics or call finish_extraction if done."))
    
    if iteration >= max_iterations:
        print(f"  ⚠ Reached max iterations ({max_iterations}), stopping extraction")
    
    return {"metrics_count": count, "current_paper": {}, "paper_md": ""}

# Routing functions
def route_after_abstract(state: GraphState) -> Literal["download_paper", "check_abstract", "create_query"]:
    """Route based on abstract check result"""
    if state.get("current_paper", {}).get("doi"):
        return "download_paper"
    elif state.get("papers", []):
        return "check_abstract"
    else:
        return "create_query"

def route_after_download(state: GraphState) -> Literal["extract_interactions", "check_abstract", "create_query"]:
    """Route based on download success"""
    if state.get("paper_md"):
        return "extract_interactions"
    elif state.get("papers", []):
        return "check_abstract"
    else:
        return "create_query"

def route_after_extraction(state: GraphState) -> Literal["check_abstract", "create_query", END]:
    """Route based on metrics count"""
    count = state.get("metrics_count", 0)
    min_count = state.get("min_metrics", 5)
    
    print(f"\n--- Risk Metrics: {count}/{min_count} ---")
    
    if count >= min_count:
        print("✓ Enough metrics found!")
        return END
    elif state.get("papers", []):
        print("→ Checking next paper")
        return "check_abstract"
    else:
        print("→ Searching for more papers")
        return "create_query"

# Build workflow
workflow = StateGraph(GraphState)

# Add nodes
workflow.add_node("create_query", create_query)
workflow.add_node("search_pubmed", search_pubmed)
workflow.add_node("filter_papers", filter_papers)
workflow.add_node("check_abstract", check_abstract)
workflow.add_node("download_paper", download_paper)
workflow.add_node("extract_interactions", extract_interactions)

# Add edges
workflow.add_edge(START, "create_query")
workflow.add_edge("create_query", "search_pubmed")
workflow.add_edge("search_pubmed", "filter_papers")
workflow.add_edge("filter_papers", "check_abstract")

workflow.add_conditional_edges(
    "check_abstract",
    route_after_abstract,
    {
        "download_paper": "download_paper",
        "check_abstract": "check_abstract",
        "create_query": "create_query"
    }
)

workflow.add_conditional_edges(
    "download_paper",
    route_after_download,
    {
        "extract_interactions": "extract_interactions",
        "check_abstract": "check_abstract",
        "create_query": "create_query"
    }
)

workflow.add_conditional_edges(
    "extract_interactions",
    route_after_extraction,
    {
        "check_abstract": "check_abstract",
        "create_query": "create_query",
        END: END
    }
)

# Compile with increased recursion limit
agent = workflow.compile()
agent = agent.with_config(recursion_limit=400)

if __name__ == "__main__":
    result = agent.invoke(
        {
            "disease_of_interest": "cardiovascular disease",
            "metrics_count": 0,
            "min_metrics": 10,
            "checked_dois": [],
            "tried_queries": []
        },
        {"recursion_limit": 400}
    )
    print(f"\n\n=== FINAL RESULT ===")
    print(f"Total risk metrics found: {result.get('metrics_count', 0)}")
    print(f"Papers checked: {len(result.get('checked_dois', []))}")
