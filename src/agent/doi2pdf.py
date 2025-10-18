import os
import re
import json
import urllib.request
import urllib.parse
from typing import Optional
from dotenv import load_dotenv
load_dotenv()


class PDFFromDOI:
    def __init__(self, output_dir: str = "pdfs", brightdata_api_key: Optional[str] = None, unpaywall_email: str = "test@google.com") -> None:
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.brightdata_api_key = brightdata_api_key or os.environ.get("BRIGHT_WEB_UNLOCKER_KEY")
        self.unpaywall_email = unpaywall_email

    def download(self, doi: str, filename: str = None) -> Optional[str]:
        path = os.path.join(self.output_dir, f"{self._sanitize_filename(filename or doi)}.pdf")
        
        # Try arXiv direct download first if it's an arXiv DOI
        if self._is_arxiv_doi(doi):
            pdf_url = self._get_arxiv_pdf_url(doi)
            if pdf_url and self._download_pdf_direct(pdf_url, path):
                self._validate_pdf(path)
                return path
        
        # Fallback to Unpaywall
        pdf_url = self._get_pdf_url_from_unpaywall(doi)
        if not pdf_url:
            raise FileNotFoundError(f"No open-access PDF found for DOI: {doi}")
        # Try Bright Data first, fallback to direct download
        if self._download_pdf_via_brightdata(pdf_url, path):
            self._validate_pdf(path)
            return path
        elif self._download_pdf_direct(pdf_url, path):
            self._validate_pdf(path)
            return path
        raise RuntimeError(f"Failed to download PDF from: {pdf_url}")

    def _get_pdf_url_from_unpaywall(self, doi: str) -> str:
        base = "https://api.unpaywall.org/v2/"
        url = f"{base}{urllib.parse.quote(doi)}?{urllib.parse.urlencode({'email': self.unpaywall_email})}"
        req = urllib.request.Request(url)
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            raise RuntimeError(f"Unpaywall lookup failed for DOI: {doi}") from e
        best = data.get("best_oa_location") or {}
        pdf_url = best.get("url_for_pdf")
        if not pdf_url:
            raise FileNotFoundError(f"No open-access PDF URL in Unpaywall response for DOI: {doi}")
        print(pdf_url)
        return pdf_url

    def _download_pdf_via_brightdata(self, pdf_url: str, out_path: str) -> bool:
        if not self.brightdata_api_key:
            return False
        body = json.dumps({"zone": "web_unlocker1", "url": pdf_url, "format": "raw"}).encode("utf-8")
        req = urllib.request.Request(
            "https://api.brightdata.com/request",
            data=body,
            headers={"Authorization": f"Bearer {self.brightdata_api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp, open(out_path, "wb") as f:
                f.write(resp.read())
            return True
        except Exception:
            return False

    def _download_pdf_direct(self, pdf_url: str, out_path: str) -> bool:
        """Direct download fallback for open-access PDFs"""
        try:
            req = urllib.request.Request(pdf_url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})
            with urllib.request.urlopen(req, timeout=30) as resp, open(out_path, "wb") as f:
                f.write(resp.read())
            return True
        except Exception:
            return False

    def _is_arxiv_doi(self, doi: str) -> bool:
        """Check if DOI is from arXiv (format: 10.48550/arXiv.XXXX)"""
        return doi.startswith("10.48550/arXiv.")
    
    def _get_arxiv_pdf_url(self, doi: str) -> str:
        """Extract arXiv ID from DOI and construct PDF URL"""
        if not self._is_arxiv_doi(doi):
            raise ValueError(f"Not an arXiv DOI: {doi}")
        arxiv_id = doi.split("arXiv.")[-1]
        return f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    
    def _validate_pdf(self, path: str) -> None:
        """Check if file is actually a PDF, raise error if not"""
        with open(path, "rb") as f:
            if f.read(5) != b"%PDF-":
                os.remove(path)
                raise RuntimeError(f"Downloaded file is HTML, not PDF (likely paywalled)")
    
    def _sanitize_filename(self, filename: str) -> str:
        return re.sub(r"[\\/*?:\"<>|]", "_", filename)

if __name__ == "__main__":
    pdf_from_doi = PDFFromDOI()
    path = pdf_from_doi.download("10.1001/jamacardio.2016.2415")
    print(path)