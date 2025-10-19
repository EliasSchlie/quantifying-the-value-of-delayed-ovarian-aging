import os
import re
import json
import urllib.request
import urllib.parse
import sys
from typing import Optional
from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from paperHTML2pdf import html_to_pdf


class PDFFromDOI:
    def __init__(self, output_dir: str = "pdfs", brightdata_api_key: Optional[str] = None, unpaywall_email: str = "test@google.com") -> None:
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.brightdata_api_key = brightdata_api_key or os.environ.get("BRIGHT_WEB_UNLOCKER_KEY")
        if not self.brightdata_api_key:
            print("Bright Data API key is not set")
        self.unpaywall_email = unpaywall_email

    def download(self, doi: str, filename: str = None) -> Optional[str]:
        path = os.path.join(self.output_dir, f"{self._sanitize_filename(filename or doi)}.pdf")
        
        # Try arXiv direct download first if it's an arXiv DOI
        if self._is_arxiv_doi(doi):
            pdf_url = self._get_arxiv_pdf_url(doi)
            if pdf_url and self._download_pdf_direct(pdf_url, path):
                return self._detect_and_fix_format(path, pdf_url)
        
        # Get URL from Unpaywall
        pdf_url = self._get_pdf_url_from_unpaywall(doi)
        if not pdf_url:
            raise FileNotFoundError(f"No open-access PDF found for DOI: {doi}")
        
        # Try Bright Data first, fallback to direct download
        if self._download_pdf_via_brightdata(pdf_url, path):
            return self._detect_and_fix_format(path, pdf_url)
        elif self._download_pdf_direct(pdf_url, path):
            return self._detect_and_fix_format(path, pdf_url)
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
            with urllib.request.urlopen(req, timeout=60) as resp:
                content = resp.read()
                if len(content) < 100:
                    print(f"Bright Data returned too little data: {len(content)} bytes")
                    return False
                with open(out_path, "wb") as f:
                    f.write(content)
            return os.path.getsize(out_path) > 100
        except Exception as e:
            print(f"Bright Data failed: {e}")
            return False

    def _download_pdf_direct(self, pdf_url: str, out_path: str) -> bool:
        """Direct download fallback for open-access PDFs"""
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "application/pdf,*/*",
                "Accept-Language": "en-US,en;q=0.9"
            }
            req = urllib.request.Request(pdf_url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                content = resp.read()
                if len(content) < 100:
                    print(f"Direct download returned too little data: {len(content)} bytes")
                    return False
                with open(out_path, "wb") as f:
                    f.write(content)
            return os.path.getsize(out_path) > 100
        except Exception as e:
            print(f"Direct download failed: {e}")
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
    
    def _extract_pdf_links(self, html_content: str, base_url: str) -> list:
        """Extract potential PDF download links from HTML"""
        from urllib.parse import urljoin
        pdf_links = []
        
        # Pattern 1: Links ending in .pdf
        pdf_urls = re.findall(r'href=["\']([^"\']*\.pdf(?:\?[^"\']*)?)["\']', html_content, re.IGNORECASE)
        pdf_links.extend(pdf_urls)
        
        # Pattern 2: Links with pdf-related paths (pdf, pdfdirect, pdfviewer, etc.)
        pdf_path_patterns = [
            r'href=["\']([^"\']*?/pdf[^/\s"\']*?/[^"\']*?)["\']',  # /pdf/, /pdfdirect/, /pdfviewer/
            r'href=["\']([^"\']*?download[^"\']*?\.pdf[^"\']*?)["\']',  # download with .pdf
            r'href=["\']([^"\']*?\?[^"\']*download[^"\']*?)["\']',  # ?download=true
        ]
        for pattern in pdf_path_patterns:
            pdf_links.extend(re.findall(pattern, html_content, re.IGNORECASE))
        
        # Pattern 3: Links with "download" text (case insensitive)
        download_links = re.findall(r'href=["\']([^"\']+?)["\'][^>]*>[\s\S]{0,50}?download[\s\S]{0,50}?</a>', 
                                    html_content, re.IGNORECASE)
        pdf_links.extend(download_links[:5])
        
        # Filter out non-document files
        exclude_exts = ('.css', '.js', '.png', '.jpg', '.jpeg', '.gif', '.svg', '.woff', '.woff2', '.ttf', '.eot', '.ico')
        pdf_links = [url for url in pdf_links if not any(url.lower().endswith(ext) for ext in exclude_exts)]
        
        # Convert relative URLs to absolute and deduplicate
        return list(set([urljoin(base_url, url) for url in pdf_links if url]))
    
    def _detect_and_fix_format(self, path: str, original_url: str = None) -> str:
        """Detect if file is PDF or HTML, extract PDF links and retry if HTML"""
        if not os.path.exists(path) or os.path.getsize(path) < 100:
            raise RuntimeError(f"Downloaded file is empty or too small: {path}")
        
        with open(path, "rb") as f:
            header = f.read(5)
        
        if header == b"%PDF-":
            return path
        
        # It's HTML - try to extract PDF download links
        html_path = path.replace(".pdf", ".html")
        os.rename(path, html_path)
        
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
    
        
        if len(html_content) < 100:
            raise RuntimeError(f"HTML content is too small ({len(html_content)} bytes)")
        
        # Determine base URL from original URL or HTML <base> tag
        base_url = original_url
        if not base_url:
            base_url_match = re.search(r'<base\s+href=["\']([^"\']+)["\']', html_content, re.IGNORECASE)
            base_url = base_url_match.group(1) if base_url_match else "https://"
        
        pdf_links = self._extract_pdf_links(html_content, base_url)
        
        if pdf_links:
            print(f"Found {len(pdf_links)} potential PDF link(s), trying...")
        
        # Try each extracted PDF link
        for i, pdf_link in enumerate(pdf_links[:5], 1):
            print(f"  Trying link {i}/{min(len(pdf_links), 5)}: {pdf_link[:80]}...")
            if self._download_pdf_via_brightdata(pdf_link, path) or self._download_pdf_direct(pdf_link, path):
                if os.path.exists(path) and os.path.getsize(path) > 100:
                    with open(path, "rb") as f:
                        if f.read(5) == b"%PDF-":
                            print(f"  ✓ Successfully downloaded PDF")
                            return path
                if os.path.exists(path):
                    os.remove(path)
        
        # Fallback: convert HTML to PDF via Markdown
        print(f"No valid PDF links found, converting HTML to PDF via Markdown...")
        pdf_path = html_to_pdf(html_path)
        return pdf_path
    
    def _sanitize_filename(self, filename: str) -> str:
        return re.sub(r"[\\/*?:\"<>|]", "_", filename)

if __name__ == "__main__":
    pdf_from_doi = PDFFromDOI()
    path = pdf_from_doi.download("10.1155/2021/6636856")
    print(path)