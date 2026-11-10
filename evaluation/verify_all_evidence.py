
import re
from pathlib import Path
from pypdf import PdfReader

PDF_PATH = Path("v4_reranking/documents/GlobalEVOutlook2025.pdf")

search_terms = {
    1: ["17 million"],
    2: ["11 million"],
    3: ["almost half", "close to 50%"],
    4: ["Europe", "stagnate", "electric car sales"],
    5: ["United States", "increasing by just 10%"],
    6: ["Brazil", "more than doubled", "125 000"],
    7: ["India", "2% in 2024"],
    8: ["over 60%", "emerging economies"],
    9: ["40% of global exports"],
    10: ["global trade of electric cars increased"],
    11: ["price of battery electric cars", "conventional cars"],
    12: ["two-thirds", "China", "conventional equivalents"],
    13: ["public charging stations", "doubled"],
    14: ["75%", "50 kilometres", "highways"],
    15: ["electric trucks", "total truck sales"],
    16: ["two to three times", "diesel truck"],
    17: ["more than 70% of global production"],
    18: ["first quarter of 2025", "20 million"],
    19: ["exceed 40%", "2030"],
    20: ["lifecycle emissions", "life cycle emissions"],
}

reader = PdfReader(str(PDF_PATH))

print(f"Total PDF pages: {len(reader.pages)}")

for question, terms in search_terms.items():
    print("\n" + "=" * 65)
    print(f"QUESTION {question}")
    print("Search terms:", ", ".join(terms))

    matches = []

    for index, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        lower_text = text.lower()

        matched = [term for term in terms if term.lower() in lower_text]

        if matched:
            # Prefer the printed page label when available.
            label = re.search(r"PAGE\s*\|\s*(\d+)", text)

            page_number = label.group(1) if label else str(index + 1)

            for term in matched:
                position = lower_text.find(term.lower())
                start = max(0, position - 180)
                end = min(len(text), position + 350)

                snippet = " ".join(text[start:end].split())

                matches.append(
                    (page_number, term, snippet)
                )

    if not matches:
        print("No matching evidence found.")
    else:
        for page, term, snippet in matches[:8]:
            print(f"\nPDF PAGE {page} | Match: {term}")
            print(snippet)

print("\n" + "=" * 65)
print("SCAN COMPLETED")
