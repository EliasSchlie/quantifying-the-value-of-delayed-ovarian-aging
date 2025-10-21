from dotenv import load_dotenv
from pubmedAPI import PubMedAPI
from agent import extract_metadata, evaluate_robis, extract_risk_metrics
import pymupdf4llm
import os
import re
from pathlib import Path
from langsmith import traceable

load_dotenv()

pubmed_api = PubMedAPI()


def extract_doi_from_markdown(md: str) -> str:
    """Extract first DOI from markdown (usually the paper's own DOI)"""
    patterns = [
        r'doi\.org/(10\.[^\s\)\]]+)',
        r'DOI:\s*(10\.[^\s\)\]]+)',
        r'doi:\s*(10\.[^\s\)\]]+)'
    ]
    for pattern in patterns:
        match = re.search(pattern, md, re.IGNORECASE)
        if match:
            return match.group(1).rstrip('.,;')
    return ""


@traceable(name="process_manual_pdfs")
def process_manual_pdfs(folder: str, disease_of_interest: str = "cardiovascular disease") -> int:
    """Process manually added PDFs from a folder
    
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
    
    print(f"\n=== PROCESSING MANUAL PDFs FROM {folder} ===")
    print(f"Found {len(pdf_files)} PDF(s)")
    
    metrics_count = 0
    
    for pdf_path in pdf_files:
        print(f"\n--- Processing: {pdf_path.name} ---")
        
        try:
            md = pymupdf4llm.to_markdown(str(pdf_path))
            print(f"✓ Extracted markdown ({len(md)} chars)")
        except Exception as e:
            print(f"✗ Failed to extract markdown: {e}")
            continue
        
        doi = extract_doi_from_markdown(md)
        if not doi:
            print("✗ No DOI found in markdown")
            continue
        print(f"✓ Found DOI: {doi}")
        
        try:
            papers = pubmed_api.search(doi, max_results=1)
            if not papers:
                print("✗ No PubMed results for DOI")
                continue
            paper = papers[0]
            print(f"✓ Retrieved metadata: {paper.get('title', 'No title')[:50]}...")
        except Exception as e:
            print(f"✗ Failed to fetch PubMed data: {e}")
            continue
        
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
    
    print(f"\n=== MANUAL PDF PROCESSING COMPLETE ===")
    print(f"Total metrics extracted: {metrics_count}")
    return metrics_count


if __name__ == "__main__":
    folder_path = "test_pdfs/cvd"
    disease = "All cause mortality"
    
    total_metrics = process_manual_pdfs(folder_path, disease)
    
    print(f"\n\n=== FINAL RESULT ===")
    print(f"Total risk metrics extracted: {total_metrics}")

