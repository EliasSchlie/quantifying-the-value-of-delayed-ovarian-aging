from docling.document_converter import DocumentConverter
from pathlib import Path
import pymupdf4llm
import re
import logging

# Suppress docling's verbose logging
logging.getLogger('docling').setLevel(logging.WARNING)


def fix_encoding_errors(docling_text: str, pymupdf_text: str) -> str:
    """
    Fix encoding errors in docling text by finding /uni... patterns and replacing
    them with the correct text from pymupdf based on surrounding context.
    """
    # Find all encoding error patterns with context, including surrounding spaces
    # Pattern: capture word/context before, spaces, the error, spaces, word/context after
    # Use word boundaries or whitespace to capture context flexibly
    encoding_pattern = r'(\S+\s+\S*|\S*)(\s*)(/uni[A-F0-9]{4}|&[a-z]+;)(\s*)(\S*\s+\S+|\S*)'

    matches = list(re.finditer(encoding_pattern, docling_text))

    # Process matches in reverse order to preserve string indices
    for match in reversed(matches):
        before_context = match.group(1)
        after_context = match.group(5)

        # Clean the context to make it easier to search in pymupdf
        before_clean = before_context.strip()
        after_clean = after_context.strip()

        # Skip if we don't have ANY context
        if not before_clean and not after_clean:
            continue

        # Build a search pattern: look for text that has similar before/after context
        # Use as much context as we have available
        if before_clean and after_clean:
            # Take last 20 chars of before and first 20 chars of after
            before_part = before_clean[-20:] if len(before_clean) > 20 else before_clean
            after_part = after_clean[:20] if len(after_clean) > 20 else after_clean
            search_pattern = re.escape(before_part) + r'(.{1,5})' + re.escape(after_part)
        elif before_clean:
            # Only have before context
            before_part = before_clean[-25:] if len(before_clean) > 25 else before_clean
            search_pattern = re.escape(before_part) + r'(.{1,5})'
        else:
            # Only have after context
            after_part = after_clean[:25] if len(after_clean) > 25 else after_clean
            search_pattern = r'(.{1,5})' + re.escape(after_part)

        # Search in pymupdf text
        pymupdf_match = re.search(search_pattern, pymupdf_text, re.IGNORECASE)

        if pymupdf_match:
            # Get the correct character(s) from pymupdf
            correct_chars = pymupdf_match.group(1).strip()

            # Only replace if the correct replacement is reasonable (1-3 chars usually)
            if 0 < len(correct_chars) <= 3:
                # Replace ONLY the error part: spaces_before + error + spaces_after
                # with just the correct character (no spaces)
                error_start = match.start(2)  # Start of spaces before error
                error_end = match.end(4)      # End of spaces after error

                docling_text = docling_text[:error_start] + correct_chars + docling_text[error_end:]

    return docling_text


def pdf_to_markdown(pdf_path: str | Path) -> str:
    """
    Convert a PDF to markdown using docling for structure/tables and pymupdf4llm
    to fix encoding errors.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Cleaned markdown text with fixed encoding errors
    """
    pdf_path = Path(pdf_path)

    # Extract with docling (better structure, tables)
    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))
    docling_markdown = result.document.export_to_markdown()

    # Extract with pymupdf4llm (better character encoding)
    pymupdf_markdown = pymupdf4llm.to_markdown(str(pdf_path))

    # Fix encoding errors in docling text using pymupdf as reference
    fixed_markdown = fix_encoding_errors(docling_markdown, pymupdf_markdown)

    return fixed_markdown


if __name__ == "__main__":
    pdf_path = Path(__file__).parent.parent.parent / "test_pdfs" / "cvd" / "1-s2.0-S0167527315306586-main.pdf"
    output_path = pdf_path.with_suffix('.md')

    markdown = pdf_to_markdown(pdf_path)

    # Write to markdown file
    with open(output_path, 'w') as f:
        f.write(markdown)

    print(f"Converted: {pdf_path.name}")
    print(f"Output: {output_path}")
