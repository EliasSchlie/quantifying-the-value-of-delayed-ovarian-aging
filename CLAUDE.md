# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an automated research pipeline for the "Agentic AI Against Aging" hackathon. The system uses LangGraph to discover medical research papers and extract risk metrics (OR, HR, RR) linking menopause timing to various health outcomes.

**Key outcomes tracked**: All Cause Mortality, Type 2 Diabetes, Cardiovascular Disease, All Cause Dementia, Osteoporosis & Fractures, Breast Cancer, Endometrial/Ovarian Cancer

## Development Commands

### Environment Setup
```bash
# Project uses uv for dependency management (Python 3.13+)
uv sync

# Activate virtual environment
source .venv/bin/activate
```

### Running the Pipeline
```bash
# Run the main agent (processes all diseases)
cd src/agent && python paperfinder.py

# Test with single disease
# Edit paperfinder.py line 656-657: set testing=True and go_through_all_diseases=False
cd src/agent && python paperfinder.py

# Process manually downloaded PDFs
cd src/agent && python -c "from manual_pdf_processor import process_manual_pdfs; process_manual_pdfs('extra_pdfs/cvd', 'cardiovascular disease')"
```

### API Testing
```bash
# Test Nebius LLM connection
python test_nebius.py

# Test PubMed API
cd src/agent && python pubmed.py

# Test DOI to PDF download
cd src/agent && python doi2pdf.py
```

### View MLflow Tracking
```bash
mlflow ui
# Opens at http://localhost:5000 to view LangGraph execution traces
```

## Architecture

### LangGraph Workflow (`src/agent/paperfinder.py`)

The core is a stateful LangGraph agent that orchestrates the entire pipeline:

**State** (`GraphState`):
- `disease_of_interest`: Target health outcome
- `query`: Current PubMed search query
- `papers`: Queue of papers to process
- `checked_dois`: Deduplication tracker (accumulated across queries)
- `tried_queries`: Query variation tracker
- `current_paper`: Paper being processed
- `paper_md`: Full-text markdown
- `metrics_count`: Total extracted metrics
- `min_metrics`, `max_papers`, `max_queries`: Stopping conditions

**Node Sequence**:
1. **create_query** (AI): Generates/adapts PubMed queries using ChatNebius (Qwen3-235B)
2. **search_pubmed** (Tool): Fetches meta-analyses via PubMed API
3. **filter_papers**: Removes already-checked DOIs
4. **check_abstract** (AI): Relevance filter using abstract
5. **download_paper** (Tool): PDF download via Unpaywall + Bright Data
6. **extract_metadata** (AI): Extracts n_of_included_studies, sample_size, geography, confounder_vars
7. **evaluate_robis** (AI): Risk-of-bias assessment using ROBIS framework (see `src/prompts/robis.md`)
8. **extract_interactions** (AI): Tool-calling loop to extract risk metrics with 95% CIs

**Routing Logic**:
- **After abstract check**: If relevant → download | if more papers → check next | else → new query or END (if max_queries reached)
- **After download**: If success → extract_metadata | if failed → track_failed_download → check next or new query
- **After metadata extraction**: If PDF valid → evaluate_robis | else → check next or new query or END
- **After ROBIS evaluation**: Continue to extract_interactions | or check next or new query or END
- **After extraction**: Check next paper | or new query | or END
- **Termination**: Triggers on `metrics_count >= min_metrics`, `len(checked_dois) >= max_papers`, or `len(tried_queries) >= max_queries`

**Tool Calls in Nodes**:
- `extract_interactions`: Uses `submit_risk_metrics(metrics: List[Dict])` and `finish_extraction()` tools
- `evaluate_robis`: Uses `submit_ROBIS_score(categorical_risk: str, quality_score: int)` tool

### Supporting Modules

**`pubmed.py`**:
- Wrapper for NCBI E-utilities API
- Returns: pmid, title, abstract, authors, journal, pub_date, doi, pmc_id, keywords

**`doi2pdf.py`**:
- Downloads PDFs via Unpaywall API + Bright Data web unlocker
- Handles arXiv papers directly
- Falls back to HTML→PDF conversion (`paperHTML2pdf.py`) if needed
- Auto-detects format (validates `%PDF-` header)

**`interaction_storage.py`**:
- CSV writer for risk metrics (`menopause_risk_metrics.csv`)
- Fields: disease, menopause_timing_definition, health_outcome, metric_type, metric_value, ci_95, reference, date_published, n_of_included_studies, sample_size, geography, confounder_vars, authors, robis_categorical_risk, robis_quality_score

**`paperHTML2pdf.py`**:
- Fallback HTML→Markdown→PDF converter using `markdownify` + `markdown-pdf`
- Used when DOI resolves to HTML instead of PDF

**`manual_pdf_processor.py`**:
- Processes manually downloaded PDFs from folder paths
- Extracts DOI from PDF markdown content
- Runs same pipeline as main agent: extract_metadata → evaluate_robis → extract_interactions
- Useful for processing PDFs from `failed_downloads.csv` or curated collections in `extra_pdfs/`

### LLM Models

Primary LLM: `ChatNebius(model="Qwen/Qwen3-235B-A22B-Instruct-2507")`
Reasoning LLM: `ChatNebius(model="deepseek-ai/DeepSeek-R1-0528")` (defined but unused in current code)

API keys required in `.env`:
- `BRIGHT_WEB_UNLOCKER_KEY`: For Bright Data PDF downloads
- Nebius API credentials (check langchain-nebius docs)

### Data Flow

1. Query generation adapts based on `tried_queries` to diversify search
2. Papers filtered by `checked_dois` to prevent reprocessing
3. Failed downloads logged to `failed_downloads.csv` for manual processing
4. All extracted metrics immediately appended to `menopause_risk_metrics.csv`
5. MLflow traces stored in `mlruns/` directory

### Output Data

**`menopause_risk_metrics.csv`**: Primary output with all extracted risk metrics
- CSV format with `csv.QUOTE_ALL` for safe field handling
- 15 columns: disease, menopause_timing_definition, health_outcome, metric_type, metric_value, ci_95, reference, date_published, n_of_included_studies, sample_size, geography, confounder_vars, authors, robis_categorical_risk, robis_quality_score
- Append-only (never overwrites existing data)

**`failed_downloads.csv`**: Tracks DOIs that couldn't be downloaded
- 2 columns: disease_of_interest, doi
- Used for manual follow-up or retry with different methods

### PDF Storage

- **pdfs/**: Successfully downloaded and processed papers
- **extra_pdfs/**: Additional PDFs (manual collection) - organized by disease subdirectories
- **test_pdfs/**: Test dataset

## Important Patterns

### Recursion Limit
The LangGraph agent uses `recursion_limit=400` to handle long paper-processing chains. If you see recursion errors, this may need adjustment at `paperfinder.py:653` (in compile step) and `paperfinder.py:671,699` (in invoke calls).

### Query Diversification
The `create_query` node varies terminology when `tried_queries` has entries. Add variations to the prompt at `paperfinder.py:56-62` if searches become too narrow.

### Metadata Extraction
The `extract_metadata` function uses structured output parsing (key: value format). If extraction fails, check the prompt format at `paperfinder.py:149-166`.

### ROBIS Evaluation
The ROBIS assessment follows the framework in `src/prompts/robis.md`. Modify this file to adjust bias evaluation criteria (domains 1-4, phase 1-2 signaling questions).

### Tool Call Iteration
Both `evaluate_robis` and `extract_interactions` use agentic loops (max 10/20 iterations) to retry tool calls. If LLMs fail to call tools, check that `llm.bind_tools([...])` includes the correct tool definitions at `paperfinder.py:228` (ROBIS) and `paperfinder.py:363` (metrics extraction).

## Configuration Notes

### Workflow Parameters
Default parameters in `paperfinder.py:662-699`:
- **Testing mode**: `min_metrics=10`, `max_papers=50`, `max_queries=30`
- **Production mode**: `min_metrics=1000` (effectively unlimited), `max_papers=300`, `max_queries=30`
- These control when the agent stops processing for each disease
- **Diseases list** at `paperfinder.py:677-685`: Hardcoded list of 7 target diseases/outcomes

### API and Processing Limits
- PubMed searches default to `max_results=500` with `meta_analysis_only=True` filter
- PDF downloads timeout after 60s (Bright Data) or 30s (direct)
- Markdown conversion via `pymupdf4llm.to_markdown()` rejects papers <7000 chars (too short)
- CSV uses `csv.QUOTE_ALL` to handle commas/newlines in metadata fields
- ROBIS evaluation: max 10 iterations (`paperfinder.py:235`)
- Metric extraction: max 20 iterations (`paperfinder.py:391`)

## Common Issues

**"No open-access PDF found"**: Paper is behind paywall and Unpaywall has no link. Check `failed_downloads.csv` for DOIs.

**HTML instead of PDF**: `doi2pdf.py` extracts PDF links from HTML or falls back to HTML→PDF conversion. This may produce lower-quality text.

**MLflow connection errors**: Ensure `mlruns/` directory exists and is writable. MLflow automatically tracks LangGraph executions.

**Metric extraction misses results**: Check that papers report 95% CIs (required by prompt). Subgroup analyses are intentionally skipped per `paperfinder.py:379`.
