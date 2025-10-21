"""
CSV storage manager for extracted risk metrics and paper metadata.

This module provides a simple interface to store risk metrics extracted from research papers
in a structured CSV format. Each row represents one risk metric (OR, HR, or RR) with
associated metadata about the study.

CSV schema:
- disease: Target health outcome category
- menopause_timing_definition: Age comparison (e.g., "<40 vs 50-55")
- health_outcome: Specific outcome measured
- metric_type: OR, HR, or RR
- metric_value: Numerical risk value
- ci_95: 95% confidence interval
- reference: DOI URL
- date_published: Publication date
- n_of_included_studies: Number of studies in meta-analysis
- sample_size: Total participants
- geography: Study regions
- confounder_vars: Adjusted variables
- authors: Paper authors
- robis_categorical_risk: Low/High/Unclear
- robis_quality_score: 0-10 quality rating

The file is created with headers if it doesn't exist, and all writes are append-only.

Usage:
    storage = InteractionStorage("results.csv")
    storage.add_risk_metric(
        disease="Cardiovascular Disease",
        menopause_timing="<45 vs >=51",
        health_outcome="CHD",
        metric_type="HR",
        value="1.45",
        ci_95="1.20-1.75",
        reference="https://doi.org/10.1234/example",
        date_published="2023-01-15",
        paper_metadata={...}
    )
"""

import csv
from pathlib import Path

class InteractionStorage:
    def __init__(self, csv_path: str = "menopause_risk_metrics.csv"):
        self.csv_path = Path(csv_path)
        self._ensure_file_exists()
    
    def _ensure_file_exists(self):
        """Create CSV file with headers if it doesn't exist"""
        if not self.csv_path.exists():
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, quoting=csv.QUOTE_ALL)
                writer.writerow([
                    'disease', 'menopause_timing_definition', 'health_outcome', 'metric_type', 'metric_value', 'ci_95', 
                    'reference', 'date_published', 'n_of_included_studies', 'sample_size', 'geography', 
                    'confounder_vars', 'authors', 'robis_categorical_risk', 'robis_quality_score'
                ])
    
    def add_risk_metric(self, disease: str, menopause_timing: str, health_outcome: str, metric_type: str, value: str, ci_95: str, 
                       reference: str, date_published: str, paper_metadata: dict = None) -> str:
        """Add a risk metric to the CSV file with automatic paper metadata"""
        metadata = paper_metadata or {}
        
        # Clean text fields to prevent CSV issues
        def clean_text(text):
            if not text:
                return ''
            # Replace problematic characters but keep the text readable
            return str(text).replace('\x00', '').replace('\r\n', ' ').replace('\n', ' ').replace('\r', ' ')
        
        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_ALL)
            writer.writerow([
                clean_text(disease),
                clean_text(menopause_timing),
                clean_text(health_outcome),
                clean_text(metric_type),
                clean_text(value),
                clean_text(ci_95),
                clean_text(reference),
                clean_text(date_published),
                clean_text(metadata.get('n_of_included_studies', '')),
                clean_text(metadata.get('sample_size', '')),
                clean_text(metadata.get('geography', '')),
                clean_text(metadata.get('confounder_vars', '')),
                clean_text(metadata.get('authors', '')),
                clean_text(metadata.get('robis_categorical_risk', '')),
                clean_text(metadata.get('robis_quality_score', ''))
            ])
        return f"Risk metric stored: {health_outcome} ({metric_type}={value}, CI={ci_95})"

