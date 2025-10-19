from langchain_nebius import ChatNebius
from dotenv import load_dotenv
from pubmed import PubMedAPI
from langchain_core.messages import SystemMessage, HumanMessage
from typing_extensions import TypedDict, Annotated
from typing import Literal
from langgraph.graph import StateGraph, START, END
from doi2pdf import PDFFromDOI
import pymupdf4llm
from interaction_storage import InteractionStorage

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
    max_papers: int
    robis_categorical_risk: str
    robis_quality_score: str

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
- Use related terms for the health outcome

Create a concise PubMed search query while focusing on not being too restrictive. Especially if many queries failed already, try to broaden the search up."""
    else:
        prompt = f"""Disease/outcome: {state['disease_of_interest']}

Create a PubMed search query to find studies linking menopause timing (age at menopause, early menopause) to this health outcome.
Focus on meta-analyses with risk metrics (OR, HR, RR). Include relevant keywords while on not being too restrictive."""
    
    response = llm.invoke([
        SystemMessage(content="You are an expert at crafting PubMed queries. You return ONLY the query string, no other text. Whatever text you return will be used as a query in the PubMed API, so it must be a valid query string.\n Your query will be wrappend like this: '({query}) AND \"meta-analysis\"[Publication Type]' to ensure it only returns meta-analyses."),
        HumanMessage(content=prompt)
    ])
    
    query = response.content.strip()
    print(f"Generated query: {query}")
    
    return {"query": query, "tried_queries": [query]}

def search_pubmed(state: GraphState) -> dict:
    """Search PubMed API"""
    print(f"\n--- Searching PubMed: {state['query']} ---")
    papers = pubmed_api.search(state['query'], max_results=500, meta_analysis_only=True)
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
        SystemMessage(content=f"You are a paper classifier that estimates a paper's relevance from it's abstract alone. \nEstimate if this paper studies menopause timing (age at menopause, early/premature menopause) and its association with {state['disease_of_interest']}. Relevant papers likely report quantitative risk metrics (OR, HR, RR) with confidence intervals. Reply 'yes' if relevant, 'no' if not."),
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

def extract_metadata(state: GraphState) -> dict:
    """Extract metadata from paper: n_of_included_studies, sample_size, geography, confounder_vars"""
    if not state["paper_md"]:
        return {}
    
    print("\n--- Extracting paper metadata ---")
    
    paper = state["current_paper"]
    
    prompt = f"""Analyze this research paper and extract the following metadata:

1. n_of_included_studies: The number of individual studies included in the meta-analysis (if applicable). Return just the number as a string (e.g., "12"). If not a meta-analysis or not reported, return "N/A".

2. sample_size: The total number of participants across all included studies. Return just the number as a string (e.g., "324567"). If not reported, return "N/A".

3. geography: The geographic focus or origins of the included studies (e.g., countries or regions). Return as a comma-separated string (e.g., "USA, UK, Europe"). If not reported, return "N/A".

4. confounder_vars: The variables adjusted for in the analysis. Return as a comma-separated string (e.g., "Age, BMI, smoking status, socioeconomic status, hormone replacement therapy"). If not reported, return "N/A".

Return your response in this exact format (one per line):
n_of_included_studies: <value>
sample_size: <value>
geography: <value>
confounder_vars: <value>

Paper:
{state['paper_md']}"""
    
    response = llm.invoke([
        SystemMessage(content="You are a research paper metadata extractor. Extract only the requested information from meta-analyses and systematic reviews. Be precise and concise."),
        HumanMessage(content=prompt)
    ])
    
    content = response.content.strip()
    print(f"Metadata extraction response:\n{content}")
    
    # Parse the response
    metadata = {}
    for line in content.split('\n'):
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            metadata[key] = value
    
    # Update current_paper with metadata
    updated_paper = {**paper}
    updated_paper.update({
        'n_of_included_studies': metadata.get('n_of_included_studies', 'N/A'),
        'sample_size': metadata.get('sample_size', 'N/A'),
        'geography': metadata.get('geography', 'N/A'),
        'confounder_vars': metadata.get('confounder_vars', 'N/A'),
        'abstract': paper.get('abstract', ''),
        'authors': paper.get('authors', ''),
        'full_text_md': state['paper_md']
    })
    
    print(f"✓ Extracted metadata: {metadata.get('n_of_included_studies', 'N/A')} studies, {metadata.get('sample_size', 'N/A')} participants")
    
    return {"current_paper": updated_paper}

def evaluate_robis(state: GraphState) -> dict:
    """Evaluate ROBIS risk of bias for the paper"""
    if not state["paper_md"]:
        return {"robis_categorical_risk": "N/A", "robis_quality_score": "N/A"}
    
    print("\n--- Evaluating ROBIS score ---")
    
    # Load ROBIS prompt
    from pathlib import Path
    robis_path = Path(__file__).parent.parent / "prompts" / "robis.md"
    with open(robis_path, 'r') as f:
        robis_prompt = f.read()
    
    from langchain_core.tools import tool
    
    robis_scores = {}
    
    @tool
    def submit_ROBIS_score(categorical_risk: str, quality_score: int) -> str:
        """Submit the ROBIS evaluation scores for the paper.
        
        Args:
            categorical_risk: Overall risk of bias rating. Must be one of: "Low", "High", or "Unclear"
            quality_score: Numeric validity score from 0-10 based on ROBIS criteria
        """
        nonlocal robis_scores
        robis_scores['categorical_risk'] = categorical_risk
        robis_scores['quality_score'] = str(quality_score)
        return f"✓ ROBIS scores submitted: Risk={categorical_risk}, Quality={quality_score}/10"
    
    llm_with_tool = llm.bind_tools([submit_ROBIS_score])
    
    messages = [
        SystemMessage(content=robis_prompt),
        HumanMessage(content=state['paper_md'])
    ]
    
    max_iterations = 10
    iteration = 0
    
    while not robis_scores and iteration < max_iterations:
        iteration += 1
        print(f"  ROBIS evaluation iteration {iteration}...")
        
        response = llm_with_tool.invoke(messages)
        messages.append(response)
        
        if hasattr(response, 'tool_calls') and response.tool_calls:
            for tool_call in response.tool_calls:
                if tool_call['name'] == 'submit_ROBIS_score':
                    try:
                        result = submit_ROBIS_score.invoke(tool_call['args'])
                        print(f"  {result}")
                        from langchain_core.messages import ToolMessage
                        messages.append(ToolMessage(content=result, tool_call_id=tool_call['id']))
                    except Exception as e:
                        print(f"  ✗ Failed to submit ROBIS scores: {e}")
                        error_msg = f"Error: {e}. Please resubmit."
                        from langchain_core.messages import ToolMessage
                        messages.append(ToolMessage(content=error_msg, tool_call_id=tool_call['id']))
        else:
            messages.append(HumanMessage(content="Please complete the ROBIS evaluation and submit your scores using submit_ROBIS_score."))
    
    if not robis_scores:
        print("  ⚠ Failed to get ROBIS scores, using N/A")
        robis_scores = {'categorical_risk': 'N/A', 'quality_score': 'N/A'}
    
    # Update current_paper with ROBIS scores
    paper = state['current_paper']
    updated_paper = {**paper, 
                    'robis_categorical_risk': robis_scores.get('categorical_risk', 'N/A'),
                    'robis_quality_score': robis_scores.get('quality_score', 'N/A')}
    
    print(f"✓ ROBIS evaluation complete: Risk={robis_scores['categorical_risk']}, Quality={robis_scores['quality_score']}")
    
    return {
        "current_paper": updated_paper,
        "robis_categorical_risk": robis_scores['categorical_risk'],
        "robis_quality_score": robis_scores['quality_score']
    }

def extract_interactions(state: GraphState) -> dict:
    """AI extracts risk metrics from paper using tool calls, looping until done"""
    if not state["paper_md"]:
        return {"metrics_count": state.get("metrics_count", 0), "current_paper": {}, "paper_md": ""}
    
    print("\n--- Extracting risk metrics ---")
    
    paper = state['current_paper']
    doi = paper.get('doi', '')
    pub_date = paper.get('pub_date', '')
    
    # Prepare paper metadata for automatic inclusion
    paper_metadata = {
        'n_of_included_studies': paper.get('n_of_included_studies', ''),
        'sample_size': paper.get('sample_size', ''),
        'geography': paper.get('geography', ''),
        'confounder_vars': paper.get('confounder_vars', ''),
        'authors': paper.get('authors', ''),
        'robis_categorical_risk': paper.get('robis_categorical_risk', ''),
        'robis_quality_score': paper.get('robis_quality_score', '')
    }
    
    extraction_complete = False
    
    from langchain_core.tools import tool
    from typing import List, Dict
    
    @tool
    def submit_risk_metrics(metrics: List[Dict[str, str]]) -> str:
        """Submit multiple extracted risk metrics from the paper in one call.
        
        Args:
            metrics: List of risk metrics, where each metric is a dict with keys:
                - menopause_timing_definition: How menopause groups were defined
                - health_outcome: Specific health outcome
                - metric_type: Type of metric (OR, HR, or RR)
                - metric_value: The numerical value
                - ci_95: 95% confidence interval
                
        Example:
            metrics = [
                {
                    "menopause_timing_definition": "Early < 45 years vs Normal 50-54 years",
                    "health_outcome": "Coronary Heart Disease",
                    "metric_type": "HR",
                    "metric_value": "1.45",
                    "ci_95": "1.20-1.75"
                },
                {
                    "menopause_timing_definition": "Early < 45 years vs Normal 50-54 years",
                    "health_outcome": "Stroke",
                    "metric_type": "HR",
                    "metric_value": "1.28",
                    "ci_95": "1.05-1.56"
                }
            ]
        """
        submitted = 0
        output = ""
        for metric in metrics:
            try:
                interaction_storage.add_risk_metric(
                    metric['menopause_timing_definition'],
                    metric['health_outcome'],
                    metric['metric_type'],
                    metric['metric_value'],
                    metric['ci_95'],
                    doi,
                    pub_date,
                    paper_metadata
                )
                output += f"✓ Stored: {metric['health_outcome']} | {metric['metric_type']}={metric['metric_value']} (CI: {metric['ci_95']})\n"
                submitted += 1
            except Exception as e:
                output += f"✗ Failed to store metric: {metric} -- Error: {e.name}\n"
        
        output += f"Successfully submitted {submitted} risk metric(s). Call finish_extraction when done."
        print(output)
        return output
    
    @tool
    def finish_extraction() -> str:
        """Call this when you have finished extracting ALL relevant risk metrics from the paper, or if there are no relevant metrics to extract."""
        nonlocal extraction_complete
        extraction_complete = True
        return "Extraction complete."
    
    llm_with_tools = llm.bind_tools([submit_risk_metrics, finish_extraction])
    
    initial_prompt = f"""Analyze this paper and extract ALL risk metrics (OR, HR, RR) linking menopause timing to health outcomes.

Target outcome: {state['disease_of_interest']}

For EACH risk metric found, extract:
1. menopause_timing_definition: How groups were defined (e.g., "Early < 45 years vs Normal 50-54 years", "per 1-year decrease in ANM")
2. health_outcome: Specific health outcome (e.g., "Ischemic Stroke", "Type 2 Diabetes", "Coronary Heart Disease")
3. metric_type: OR, HR, or RR
4. metric_value: The numerical value (e.g., "1.45", "2.1")
5. ci_95: 95% confidence interval (e.g., "1.20-1.75", "1.8-2.4")

CRITICAL REQUIREMENTS:
- MUST have 95% CI reported (skip metrics without CI)
- Submit ALL metrics at once using submit_risk_metrics with a list of all metrics
- Extract from main results AND subgroup analyses if present
- When done, call finish_extraction

Paper:
{state['paper_md']}"""
    
    messages = [
        SystemMessage(content="You are a scientific paper analyzer. Extract ALL risk metrics (OR, HR, RR) with their 95% CI. Submit all metrics in one call using submit_risk_metrics with a list. Call finish_extraction when done."),
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
            print(response.tool_calls)
            for i, tc in enumerate(response.tool_calls):
                print(f"    Tool call {i+1}: {tc.get('name', 'unknown')} - args keys: {list(tc.get('args', {}).keys())}")
            
            tool_messages = []
            for tool_call in response.tool_calls:
                tool_name = tool_call['name']
                
                if tool_name == 'submit_risk_metrics':
                    try:
                        args = tool_call['args']
                        metrics_list = args.get('metrics', [])
                        result = submit_risk_metrics.invoke(args)
                        count += len(metrics_list)
                        tool_messages.append({
                            "role": "tool",
                            "content": result,
                            "tool_call_id": tool_call['id']
                        })
                    except Exception as e:
                        print(f"  ✗ Failed to submit metrics: {e}")
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
            messages.append(HumanMessage(content="Submit all metrics using the submit_risk_metrics tool or call the finish_extraction tool if done. (answering in XML is not enough! You need to use the tool calls!)"))
    
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

def route_after_download(state: GraphState) -> Literal["extract_metadata", "check_abstract", "create_query"]:
    """Route based on download success"""
    if state.get("paper_md"):
        return "extract_metadata"
    elif state.get("papers", []):
        return "check_abstract"
    else:
        return "create_query"

def route_after_metadata(state: GraphState) -> Literal["evaluate_robis", "check_abstract", "create_query"]:
    """Route after metadata extraction"""
    if state.get("paper_md") and state.get("current_paper", {}).get("doi"):
        return "evaluate_robis"
    elif state.get("papers", []):
        return "check_abstract"
    else:
        return "create_query"

def route_after_robis(state: GraphState) -> Literal["extract_interactions", "check_abstract", "create_query"]:
    """Route after ROBIS evaluation"""
    if state.get("paper_md") and state.get("current_paper", {}).get("doi"):
        return "extract_interactions"
    elif state.get("papers", []):
        return "check_abstract"
    else:
        return "create_query"

def route_after_extraction(state: GraphState) -> Literal["check_abstract", "create_query", END]:
    """Route based on metrics count or max papers checked"""
    count = state.get("metrics_count", 0)
    min_count = state.get("min_metrics", 5)
    papers_checked = len(state.get("checked_dois", []))
    max_papers = state.get("max_papers", float('inf'))
    
    print(f"\n--- Risk Metrics: {count}/{min_count} | Papers Checked: {papers_checked}/{max_papers} ---")
    
    if count >= min_count:
        print("✓ Enough metrics found!")
        return END
    elif papers_checked >= max_papers:
        print(f"✓ Max papers limit reached ({max_papers})")
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
workflow.add_node("extract_metadata", extract_metadata)
workflow.add_node("evaluate_robis", evaluate_robis)
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
        "extract_metadata": "extract_metadata",
        "check_abstract": "check_abstract",
        "create_query": "create_query"
    }
)

workflow.add_conditional_edges(
    "extract_metadata",
    route_after_metadata,
    {
        "evaluate_robis": "evaluate_robis",
        "check_abstract": "check_abstract",
        "create_query": "create_query"
    }
)

workflow.add_conditional_edges(
    "evaluate_robis",
    route_after_robis,
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
    testing = True
    go_through_all_diseases = False


    if testing:
        result = agent.invoke(
            {
                "disease_of_interest": "cardiovascular disease",
                "metrics_count": 0,
                "min_metrics": 10,
                "max_papers": 50,
                "checked_dois": [],
                "tried_queries": []
            },
            {"recursion_limit": 400}
        )
        print(f"\n\n=== FINAL RESULT ===")
        print(f"Total risk metrics found: {result.get('metrics_count', 0)}")
        print(f"Papers checked: {len(result.get('checked_dois', []))}")
    
    diseases = ["cardiovascular disease", "stroke", "type 2 diabetes", "breast cancer", "prostate cancer", "ovarian cancer", "endometrial cancer", "uterine cancer", "pancreatic cancer", "lung cancer", "colorectal cancer", "esophageal cancer", "gastric cancer", "liver cancer", "gallbladder cancer", "thyroid cancer", "melanoma", "multiple myeloma", "leukemia", "lymphoma", "multiple myeloma", "leukemia", "lymphoma"]

    if go_through_all_diseases:
        for disease in diseases:
            print(f"\n\n=== STARTING for disease: {disease} ===")
            result = agent.invoke(
                {
                    "disease_of_interest": disease,
                    "metrics_count": 0,
                    "min_metrics": 1000, # High enough to never trigger
                    "max_papers": 300,
                    "checked_dois": [],
                    "tried_queries": []
                },
                {"recursion_limit": 400}
            )
            print(f"\n\n=== RESULT for disease: {disease} ===")
            print(f"Total risk metrics found: {result.get('metrics_count', 0)}")
            print(f"Papers checked: {len(result.get('checked_dois', []))}")

