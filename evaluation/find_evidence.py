
from pypdf import PdfReader
from pathlib import Path

pdf_path = Path(
    "v4_reranking/documents/GlobalEVOutlook2025.pdf"
)

reader = PdfReader(str(pdf_path))

keywords = [
    "17 million",
    "11 million",
    "almost half",
    "Europe",
    "United States",
    "Brazil",
    "India",
    "60%",
    "40% of global exports",
    "global trade",
]

for keyword in keywords:
    print("\n" + "=" * 60)
    print("SEARCH:", keyword)

    found = False

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if keyword.lower() in text.lower():
            found = True
            print(f"\nPDF page: {page_number}")
            
            lines = text.splitlines()

            for i, line in enumerate(lines):
                if keyword.lower() in line.lower():
                    start = max(0, i - 2)
                    end = min(len(lines), i + 3)

                    print("\n".join(lines[start:end]))
                    print("-" * 30)

    if not found:
        print("Keyword not found.")
