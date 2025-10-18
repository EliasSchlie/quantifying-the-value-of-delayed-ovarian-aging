this is a project for the "Agentic AI Against Aging" hackathon


## Implementation
### Paper Discovery & Risk Metric Extraction Pipeline

The system uses a LangGraph workflow to automatically find research papers and extract risk metrics linking menopause timing to health outcomes.

```mermaid
flowchart TD
    START([START]) --> CQ[create_query]:::ai
    CQ --> SP[search_pubmed]:::tool
    SP --> FP[filter_papers]
    FP --> CA[check_abstract]:::ai
    
    CA --> R1{route_after_abstract}
    R1 -->|Relevant paper found| DP[download_paper]:::tool
    R1 -->|Not relevant, more papers| CA
    R1 -->|No papers left| CQ
    
    DP --> R2{route_after_download}
    R2 -->|PDF→MD success| EI[extract_interactions]:::ai
    R2 -->|Failed, more papers| CA
    R2 -->|Failed, no papers| CQ
    
    EI --> R3{route_after_extraction}
    R3 -->|metrics >= min_metrics| END([END]):::success
    R3 -->|Need more, have papers| CA
    R3 -->|Need more, no papers| CQ
    
    %% Styling
    classDef ai fill:#e1f5e1,stroke:#4caf50,stroke-width:2px
    classDef tool fill:#e3f2fd,stroke:#2196f3,stroke-width:2px
    classDef success fill:#c8e6c9,stroke:#4caf50,stroke-width:3px
```

**Nodes:**
- **create_query** (AI): Generates PubMed search queries, adapts if previous queries exhausted
- **search_pubmed** (Tool): Fetches up to 100 papers from PubMed API
- **filter_papers**: Removes already-checked DOIs to avoid duplicates
- **check_abstract** (AI): Evaluates abstract relevance (yes/no decision)
- **download_paper** (Tool): Downloads PDF via DOI → converts to markdown using `pymupdf4llm`
- **extract_interactions** (AI): Uses tool calls to extract risk metrics (OR/HR/RR with 95% CI)

**Routing Logic:**
- **route_after_abstract**: If relevant → download | if more papers → check next | else → new query
- **route_after_download**: If PDF→MD success → extract | if more papers → check next | else → new query  
- **route_after_extraction**: If enough metrics → END | if more papers → check next | else → new query

**State Tracking:**
- `checked_dois`: Prevents re-processing papers
- `tried_queries`: Enables query diversification
- `metrics_count`: Tracks progress toward `min_metrics` goal