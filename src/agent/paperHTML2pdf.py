from markdownify import markdownify as md
from markdown_pdf import MarkdownPdf, Section
import re
from bs4 import BeautifulSoup


def html_to_pdf(html_path: str) -> str:
    """Convert HTML file to PDF via Markdown intermediate"""
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    
    soup = BeautifulSoup(html_content, 'html.parser')
    for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'iframe', 'noscript']):
        tag.decompose()
    
    article = soup.find('article') or soup.find('main') or soup.find('body') or soup
    markdown = md(str(article), strip=['a'])
    markdown = re.sub(r'!\[.*?\]\(.*?\)', '', markdown)
    markdown = re.sub(r'\n{3,}', '\n\n', markdown)
    
    pdf_path = html_path.replace(".html", ".pdf")
    pdf = MarkdownPdf(toc_level=0, optimize=False)
    pdf.add_section(Section(markdown, toc=False))
    pdf.save(pdf_path)
    
    return pdf_path


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        result = html_to_pdf(sys.argv[1])
        print(result)
    else:
        print("Usage: python paperHTML2pdf.py <html_file_path>")
