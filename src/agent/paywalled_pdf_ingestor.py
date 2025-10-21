"""
Batch processor for paywalled/manually downloaded PDFs.

This module handles PDFs that couldn't be automatically downloaded via the main workflow in `agent.py`,
typically because they're behind paywalls or require institutional access. It processes
PDFs that were manually obtained and placed in the closed_access_pdfs/ folder.

Key features:
1. Processes PDFs from disk (no DOI download needed)
2. Uses AI agent to search PubMed and match PDF to metadata
3. Extracts metadata, evaluates ROBIS quality, and extracts risk metrics
4. Reuses the same extraction pipeline as the main agent workflow in `agent.py`

The PubMed matching agent:
- Analyzes the PDF content to identify title and authors
- Crafts search queries to find the paper in PubMed
- Compares results to confirm correct match
- Retrieves complete metadata (DOI, dates, etc.)

Folder structure:
    closed_access_pdfs/
        all_cause_dementia/
            paper1.pdf
            paper2.pdf
        cardiovascular_disease/
            paper3.pdf
        ...

Each subfolder name is used as the disease_of_interest for metrics extraction.

Usage:
    total_metrics = process_paywalled_pdfs("closed_access_pdfs/cardiovascular_disease", 
                                          "Cardiovascular Disease")
    
    # Or process all subfolders:
    python paywalled_pdf_ingestor.py  # (with closed_access_pdfs=True)
"""

from dotenv import load_dotenv
from pubmedAPI import PubMedAPI
from agent import extract_metadata, evaluate_robis, extract_risk_metrics
from pdf2md import pdf_to_markdown
from langchain_nebius import ChatNebius
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
import os
import re
import logging
from pathlib import Path
from langsmith import traceable

load_dotenv()

# Suppress HTTP request logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

pubmed_api = PubMedAPI()
llm = ChatNebius(model="Qwen/Qwen3-235B-A22B-Instruct-2507")


@tool
def search_pubmed(query: str) -> str:
    """
    Search PubMed with a freeform query and return the top result.

    Args:
        query: Any PubMed search query (title, authors, DOI, keywords, etc.)

    Returns:
        JSON string with a single paper metadata dict, or empty dict if no results
    """
    import json
    try:
        papers = pubmed_api.search(query, max_results=1)
        return json.dumps(papers[0] if papers else {})
    except:
        return json.dumps({})


# Global variable to store last search result
_last_search_result = None


@tool
def confirm_paper_metadata() -> str:
    """
    Confirm that the last search result is the correct paper.
    
    Returns:
        Confirmation message
    """
    global _last_search_result
    
    if _last_search_result:
        return "Paper confirmed as correct match."
    else:
        return "No search result to confirm."


@traceable(name="find_paper_metadata_agent")
def find_paper_metadata_via_agent(markdown: str) -> dict:
    """
    Use an LLM agent to search PubMed and find paper metadata.

    Args:
        markdown: The full paper markdown text

    Returns:
        Paper metadata dict if found and confirmed, empty dict otherwise
    """
    global _last_search_result
    _last_search_result = None

    system_prompt = """You are a research paper metadata extraction agent. Your task is to:
1. Analyze the paper markdown to understand its title, authors, and content
2. Craft PubMed search queries to find the paper's metadata
3. Review each search result and either confirm it or try a different query

Process:
- Call search_pubmed with a freeform query (e.g., title, author names, DOI, keywords)
- You'll get back ONE paper result (or empty dict if no match)
- Review if it matches the paper in the markdown (compare title, authors)
- If it matches: call confirm_paper_metadata() to confirm
- If it doesn't match: try a different search query
- Continue until you find the right paper or exhaust reasonable search attempts

Tips:
- Start with the most specific query (e.g., full title in quotes)
- If no match, try variations (partial title, first author + keywords)
- Compare titles allowing for minor formatting differences
- Check if authors match"""

    user_prompt = f"""Find this paper's metadata in PubMed by crafting search queries.

Paper markdown (first 3000 chars):
{markdown[:3000]}"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]

    tools = [search_pubmed, confirm_paper_metadata]
    llm_with_tools = llm.bind_tools(tools)

    # Run agent loop (max 10 iterations)
    for _ in range(10):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        # Check if agent is done
        if not response.tool_calls:
            return {}

        # Execute tool calls
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            if tool_name == "search_pubmed":
                result = search_pubmed.invoke(tool_args)
                import json
                _last_search_result = json.loads(result) if result != "{}" else None
                messages.append(ToolMessage(
                    content=result,
                    tool_call_id=tool_call["id"]
                ))
            elif tool_name == "confirm_paper_metadata":
                result = confirm_paper_metadata.invoke(tool_args)
                messages.append(ToolMessage(
                    content=result,
                    tool_call_id=tool_call["id"]
                ))
                # Return the confirmed paper
                if _last_search_result:
                    return _last_search_result

    return {}


@traceable(name="process_paywalled_pdfs")
def process_paywalled_pdfs(folder: str, disease_of_interest: str = "cardiovascular disease") -> int:
    """Process paywalled PDFs from a folder
    
    Args:
        folder: Path to folder containing PDFs to process
        disease_of_interest: The health outcome to focus on
        
    Returns:
        Total number of metrics extracted
    """
    if not folder or not os.path.exists(folder):
        print(f"✗ Folder not found: {folder}")
        return 0
    
    pdf_files = list(Path(folder).rglob("*.pdf"))
    if not pdf_files:
        print(f"\n--- No PDFs found in {folder} ---")
        return 0
    
    print(f"\n=== PROCESSING PAYWALLED PDFs FROM {folder} ===")
    print(f"Found {len(pdf_files)} PDF(s)")
    
    metrics_count = 0
    
    for pdf_path in pdf_files:
        print(f"\n--- Processing: {pdf_path.name} ---")
        
        try:
            md = pdf_to_markdown(pdf_path)
            print(f"✓ Extracted markdown ({len(md)} chars)")
        except Exception as e:
            print(f"✗ Failed to extract markdown: {e}")
            continue
        
        paper = find_paper_metadata_via_agent(md)
        if not paper:
            print("✗ Agent could not find paper in PubMed")
            continue
        print(f"✓ Agent found paper: {paper.get('title', 'No title')[:50]}...")
        
        state = {
            "disease_of_interest": disease_of_interest,
            "current_paper": paper,
            "paper_md": md,
            "metrics_count": metrics_count,
            "min_metrics": 0,
            "robis_categorical_risk": "",
            "robis_quality_score": ""
        }
        
        # Step 1: Extract metadata
        result = extract_metadata(state)
        state.update(result)
        
        # Step 2: Evaluate ROBIS
        result = evaluate_robis(state)
        state.update(result)
        
        # Step 3: Extract interactions
        result = extract_risk_metrics(state)
        metrics_count = result["metrics_count"]
        print(f"✓ Paper processed. Total metrics: {metrics_count}")
    
    print(f"\n=== PAYWALLED PDF PROCESSING COMPLETE ===")
    print(f"Total metrics extracted: {metrics_count}")
    return metrics_count


if __name__ == "__main__":

    testing = False
    closed_access_pdfs = False
    re_run_open_access_pdfs = False
    
    if testing:
        folder_path = "test_pdfs/cvd"
        disease = "cardiovascular disease"
        total_metrics = process_paywalled_pdfs(folder_path, disease)

    if True:
        folder_path = "closed_access_pdfs/osteoporosis_fractures"
        disease = "osteoporosis and fractures"
        total_metrics = process_paywalled_pdfs(folder_path, disease)

        folder_path = "closed_access_pdfs/ovarian_cancer"
        disease = "ovarian cancer"
        total_metrics = process_paywalled_pdfs(folder_path, disease)

        folder_path = "closed_access_pdfs/type_2_diabetes"
        disease = "type 2 diabetes"
        total_metrics = process_paywalled_pdfs(folder_path, disease)

    if closed_access_pdfs:
        # Process each subfolder in closed_access_pdfs with its own disease name
        closed_access_pdfs_path = Path("closed_access_pdfs")
        total_metrics = 0
        
        if closed_access_pdfs_path.exists():
            subfolders = [f for f in closed_access_pdfs_path.iterdir() if f.is_dir()]
            
            for subfolder in sorted(subfolders):
                # Convert folder name to disease name (e.g., "all_cause_dementia" -> "All Cause Dementia")
                disease = subfolder.name.replace("_", " ").title()
                
                print(f"\n{'='*60}")
                print(f"Processing disease category: {disease}")
                print(f"{'='*60}")
                
                metrics = process_paywalled_pdfs(str(subfolder), disease)
                print(f"Metrics extracted for {disease}: {metrics}")
                total_metrics += metrics
        else:
            print(f"✗ Folder not found: closed_access_pdfs")
    
    elif re_run_open_access_pdfs:
        folder_path = "pdfs"
        disease = "All cause mortality, Type 2 Diabetes, Cardiovascular Disease, All Cause Dementia, Osteoporosis & Fractures, Breast Cancer, Endometrial/ Ovarian Cancer"
        total_metrics = process_paywalled_pdfs(folder_path, disease)
    
    print(f"\n\n=== FINAL RESULT ===")
    print(f"Total risk metrics extracted: {total_metrics}")

