"""
PubMed E-utilities API client for searching and fetching research papers.

This module provides a simple interface to the NCBI PubMed database via E-utilities API.
It handles:
- Searching PubMed with custom queries
- Filtering for meta-analyses
- Fetching detailed paper metadata (title, abstract, authors, DOI, etc.)
- Batch processing for efficient API usage

The API is used without authentication but follows NCBI's usage guidelines
by including email and tool name in requests.

Usage:
    api = PubMedAPI()
    papers = api.search("menopause timing cardiovascular disease", 
                       max_results=100, 
                       meta_analysis_only=True)
    
    for paper in papers:
        print(paper['title'], paper['doi'])
"""

import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Optional


class PubMedAPI:
    def __init__(self, email: str = "test@google.com", tool: str = "research_agent"):
        self.email = email
        self.tool = tool
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def search(self, query: str, max_results: int = 100, meta_analysis_only: bool = False) -> list[dict]:
        if meta_analysis_only:
            query = f'({query}) AND "meta-analysis"[Publication Type]'
        pmids = self._search_pmids(query, max_results)
        if not pmids:
            return []
        return self._fetch_details(pmids)

    def _search_pmids(self, query: str, max_results: int) -> list[str]:
        params = urllib.parse.urlencode({
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "xml",
            "email": self.email,
            "tool": self.tool
        })
        url = f"{self.base_url}/esearch.fcgi?{params}"
        with urllib.request.urlopen(url, timeout=30) as resp:
            root = ET.fromstring(resp.read().decode("utf-8"))
        return [id_elem.text for id_elem in root.findall(".//Id")]

    def _fetch_details(self, pmids: list[str]) -> list[dict]:
        batch_size = 200
        all_articles = []
        for i in range(0, len(pmids), batch_size):
            batch = pmids[i:i + batch_size]
            params = urllib.parse.urlencode({
                "db": "pubmed",
                "id": ",".join(batch),
                "retmode": "xml",
                "email": self.email,
                "tool": self.tool
            })
            url = f"{self.base_url}/efetch.fcgi?{params}"
            with urllib.request.urlopen(url, timeout=60) as resp:
                root = ET.fromstring(resp.read().decode("utf-8"))
            all_articles.extend(root.findall(".//PubmedArticle"))
        return [self._parse_article(art) for art in all_articles]

    def _parse_article(self, article: ET.Element) -> dict:
        medline = article.find(".//MedlineCitation")
        pmid = self._get_text(medline, ".//PMID")
        art = medline.find(".//Article")
        
        return {
            "pmid": pmid,
            "title": self._get_text(art, ".//ArticleTitle"),
            "abstract": self._get_abstract(art),
            "authors": self._get_authors(art),
            "journal": self._get_text(art, ".//Journal/Title"),
            "journal_abbr": self._get_text(art, ".//Journal/ISOAbbreviation"),
            "pub_date": self._get_pub_date(art),
            "doi": self._get_article_id(article, "doi"),
            "pmc_id": self._get_article_id(article, "pmc"),
            "keywords": self._get_keywords(medline)
        }

    def _get_text(self, elem: Optional[ET.Element], path: str) -> str:
        if elem is None:
            return ""
        found = elem.find(path)
        return found.text if found is not None and found.text else ""

    def _get_abstract(self, art: ET.Element) -> str:
        texts = [e.text for e in art.findall(".//AbstractText") if e.text]
        return " ".join(texts)

    def _get_authors(self, art: ET.Element) -> list[str]:
        authors = []
        for author in art.findall(".//Author"):
            last = self._get_text(author, ".//LastName")
            first = self._get_text(author, ".//ForeName")
            if last:
                authors.append(f"{last}, {first}" if first else last)
        return authors

    def _get_pub_date(self, art: ET.Element) -> str:
        date = art.find(".//Journal/JournalIssue/PubDate")
        if date is None:
            return ""
        year = self._get_text(date, ".//Year")
        month = self._get_text(date, ".//Month")
        day = self._get_text(date, ".//Day")
        return "-".join(filter(None, [year, month, day]))

    def _get_article_id(self, article: ET.Element, id_type: str) -> str:
        pubmed_data = article.find(".//PubmedData/ArticleIdList")
        if pubmed_data is not None:
            for id_elem in pubmed_data.findall(".//ArticleId"):
                if id_elem.get("IdType") == id_type:
                    return id_elem.text or ""
        return ""

    def _get_keywords(self, medline: ET.Element) -> list[str]:
        return [kw.text for kw in medline.findall(".//Keyword") if kw.text]


if __name__ == "__main__":
    api = PubMedAPI()
    papers = api.search('("menopause timing" OR "age at menopause" OR "early menopause" OR premature menopause OR postmenopause OR"natural menopause") AND ("cardiovascular disease"[mh] OR"cardiovascular system"[mh] OR "coronary heart disease" OR"coronary artery disease" OR "myocardial infarction" OR "heart disease" OR stroke OR CVD)', max_results=3, meta_analysis_only=True)
    for i, paper in enumerate(papers, 1):
        print(f"\n{i}. {paper['title']}")
        print(f"   Authors: {', '.join(paper['authors'][:3])}")
        print(f"   Journal: {paper['journal']} ({paper['pub_date']})")
        print(f"   DOI: {paper['doi']}")
        print(f"   PMID: {paper['pmid']}")
        print(f"   Abstract: {paper['abstract'][:150]}..." if len(paper['abstract']) > 150 else f"   Abstract: {paper['abstract']}")

