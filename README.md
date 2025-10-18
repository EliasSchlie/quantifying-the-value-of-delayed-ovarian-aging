this is a project for the "Agentic AI Against Aging" hackathon


## Implementation
### Paper Discovery & Risk Metric Extraction Pipeline

The system uses a LangGraph workflow to automatically find research papers and extract risk metrics linking menopause timing to health outcomes.

```mermaid
flowchart TD
    START([START]) --> CQ[create_query]:::ai
    CQ -->|Generate PubMed query| SP[search_pubmed]:::tool
    SP -->|Fetch papers| FP[filter_papers]
    FP -->|Remove checked DOIs| CA[check_abstract]:::ai
    
    CA -->|Abstract relevant?| R1{Has DOI?}
    R1 -->|Yes| DP[download_paper]:::tool
    R1 -->|No more papers| CQ
    R1 -->|Check next| CA
    
    DP -->|Convert to markdown| R2{Success?}
    R2 -->|Yes| EI[extract_interactions]:::ai
    R2 -->|Failed, more papers| CA
    R2 -->|Failed, no papers| CQ
    
    EI -->|Extract OR/HR/RR with CI| R3{Enough metrics?}
    R3 -->|metrics >= min_metrics| DONE([END]):::success
    R3 -->|More papers to check| CA
    R3 -->|Need more papers| CQ
    
    %% Styling
    classDef ai fill:#e1f5e1,stroke:#4caf50,stroke-width:2px
    classDef tool fill:#e3f2fd,stroke:#2196f3,stroke-width:2px
    classDef success fill:#c8e6c9,stroke:#4caf50,stroke-width:3px
```

**Key Components:**
- **create_query** (AI): Generates PubMed search queries for menopause timing × health outcomes
- **search_pubmed** (Tool): Fetches up to 100 papers from PubMed API
- **filter_papers**: Removes already-checked DOIs to avoid duplicates
- **check_abstract** (AI): Evaluates abstract relevance before downloading
- **download_paper** (Tool): Downloads PDF via DOI and converts to markdown
- **extract_interactions** (AI): Extracts risk metrics (OR/HR/RR) with 95% CI using tool calls

**Loop Behavior:**
- Continues until `metrics_count >= min_metrics`
- Tries different queries if current search exhausted
- Tracks checked DOIs and tried queries to avoid repetition