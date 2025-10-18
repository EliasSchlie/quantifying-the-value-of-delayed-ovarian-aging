import csv
from pathlib import Path

class InteractionStorage:
    def __init__(self, csv_path: str = "menopause_risk_metrics.csv"):
        self.csv_path = Path(csv_path)
        self._ensure_file_exists()
    
    def _ensure_file_exists(self):
        """Create CSV file with headers if it doesn't exist"""
        if not self.csv_path.exists():
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['menopause_timing_definition', 'health_outcome', 'metric_type', 'metric_value', 'ci_95', 'reference', 'date_published'])
    
    def add_risk_metric(self, menopause_timing: str, health_outcome: str, metric_type: str, value: str, ci_95: str, reference: str, date_published: str) -> str:
        """Add a risk metric to the CSV file"""
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([menopause_timing, health_outcome, metric_type, value, ci_95, reference, date_published])
        return f"Risk metric stored: {health_outcome} ({metric_type}={value}, CI={ci_95})"

