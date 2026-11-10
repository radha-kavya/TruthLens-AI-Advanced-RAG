
import os
import re
import hashlib
from pathlib import Path

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from sentence_transformers import CrossEncoder


# --------------------------------------------------
# 1. LOAD ENVIRONMENT VARIABLES
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError("GROQ_API_KEY is missing in .env")


# --------------------------------------------------
# 2. LOAD PDF DOCUMENTS
# --------------------------------------------------

documents_path = BASE_DIR / "documents"

pdf_files = sorted(documents_path.glob("*.pdf"))

if not pdf_files:
    raise FileNotFoundError("No PDF files found.")

documents = []

print("\nLoading PDF documents...")

for pdf_file in pdf_files:

    print("Loading:", pdf_file.name)

    loader = PyPDFLoader(str(pdf_file))
    pages = loader.load()

    for page in pages:
        page.metadata["source"] = pdf_file.name
        page.metadata["document_type"] = "Research Report"

    documents.extend(pages)

print("Total PDFs:", len(pdf_files))
print("Total pages:", len(documents))


# --------------------------------------------------
# 3. SPLIT DOCUMENTS
# --------------------------------------------------

print("\nSplitting documents...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)

chunks = text_splitter.split_documents(documents)

print("Total chunks before duplicate removal:", len(chunks))


# --------------------------------------------------
# 4. REMOVE EXACT DUPLICATE CHUNKS
# --------------------------------------------------

print("\nRemoving duplicate chunks...")

unique_chunks = []
seen_chunks = set()

for chunk in chunks:

    source = chunk.metadata.get("source", "Unknown")
    page = chunk.metadata.get("page", "Unknown")
    content = chunk.page_content.strip()

    duplicate_key = (source, page, content)

    if duplicate_key not in seen_chunks:
        seen_chunks.add(duplicate_key)
        unique_chunks.append(chunk)

chunks = unique_chunks

print("Chunks after duplicate removal:", len(chunks))


# --------------------------------------------------
# 5. GENERATE UNIQUE CHUNK IDS
# --------------------------------------------------

print("\nGenerating unique chunk IDs...")

chunk_ids = []

for chunk in chunks:

    source = chunk.metadata.get("source", "Unknown")
    page = chunk.metadata.get("page", "Unknown")
    content = chunk.page_content

    unique_text = f"{source}_{page}_{content}"

    chunk_id = hashlib.sha256(
        unique_text.encode("utf-8")
    ).hexdigest()

    chunk_ids.append(chunk_id)

print("Unique chunk IDs generated:", len(chunk_ids))


# --------------------------------------------------
# 6. LOAD HUGGING FACE EMBEDDINGS
# --------------------------------------------------

print("\nLoading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

print("Embedding model loaded!")


# --------------------------------------------------
# 7. CREATE CHROMADB
# --------------------------------------------------

print("\nCreating ChromaDB...")

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    ids=chunk_ids,
    collection_name="truthlens_v4_clean",
    persist_directory=str(BASE_DIR / "chroma_db")
)

print("ChromaDB created successfully!")
print("Indexed chunks:", len(chunks))


# --------------------------------------------------
# 8. RETRIEVE TOP 20 CHUNKS
# --------------------------------------------------

retriever = vector_store.as_retriever(
    search_kwargs={"k": 20}
)


# --------------------------------------------------
# 9. LOAD CROSS-ENCODER
# --------------------------------------------------

print("\nLoading Cross-Encoder model...")

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

print("Cross-Encoder loaded successfully!")


# --------------------------------------------------
# 10. GET USER CLAIM
# --------------------------------------------------

claim = input("\nEnter a claim to analyze: ").strip()

if not claim:
    raise ValueError("Please enter a valid claim.")


# --------------------------------------------------
# 11. RETRIEVE DOCUMENTS
# --------------------------------------------------

print("\nRetrieving top 20 chunks...")

retrieved_docs = retriever.invoke(claim)

if not retrieved_docs:
    print("No relevant documents found.")
    print("Verdict: Insufficient Evidence")
    raise SystemExit

print("Retrieved chunks:", len(retrieved_docs))


# --------------------------------------------------
# 12. CROSS-ENCODER RERANKING
# --------------------------------------------------

print("\nReranking documents...")

pairs = [
    (claim, doc.page_content)
    for doc in retrieved_docs
]

scores = reranker.predict(pairs)

ranked_results = sorted(
    zip(scores, retrieved_docs),
    key=lambda item: float(item[0]),
    reverse=True
)

print("Reranking completed!")


# --------------------------------------------------
# 13. RELEVANCE FILTERING + SOURCE DIVERSITY
# --------------------------------------------------

print("\nFiltering evidence by relevance...")

top_docs = []

seen_pages = set()
source_counts = {}

MAX_EVIDENCE_PAGES = 5
MAX_PAGES_PER_SOURCE = 2

highest_score = max(
    float(score) for score, doc in ranked_results
)

RELEVANCE_MARGIN = 3.0

relevance_threshold = highest_score - RELEVANCE_MARGIN

print(f"Highest score: {highest_score:.4f}")
print(f"Relevance threshold: {relevance_threshold:.4f}")

relevant_results = [
    (score, doc)
    for score, doc in ranked_results
    if float(score) >= relevance_threshold
]

print("Relevant chunks:", len(relevant_results))


# FIRST PASS: Maximum 2 pages per PDF

for score, doc in relevant_results:

    source = doc.metadata.get("source", "Unknown")
    page = doc.metadata.get("page", "Unknown")

    page_key = (source, page)

    if page_key in seen_pages:
        continue

    if source_counts.get(source, 0) >= MAX_PAGES_PER_SOURCE:
        continue

    seen_pages.add(page_key)
    top_docs.append(doc)

    source_counts[source] = source_counts.get(source, 0) + 1

    if len(top_docs) == MAX_EVIDENCE_PAGES:
        break


# SECOND PASS: Fill remaining slots

if len(top_docs) < MAX_EVIDENCE_PAGES:

    for score, doc in relevant_results:

        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "Unknown")

        page_key = (source, page)

        if page_key in seen_pages:
            continue

        seen_pages.add(page_key)
        top_docs.append(doc)

        source_counts[source] = source_counts.get(source, 0) + 1

        if len(top_docs) == MAX_EVIDENCE_PAGES:
            break


print("Evidence pages before quality validation:", len(top_docs))

print("\nEvidence distribution:")

for source, count in source_counts.items():
    print(f"{source}: {count} pages")


# --------------------------------------------------
# 14. CREATE GROQ LLM
# --------------------------------------------------

def create_llm():

    print("\nConnecting to Groq...")

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        groq_api_key=groq_api_key,
        temperature=0,
        max_retries=1
    )

    print("Groq selected!")

    return llm


# --------------------------------------------------
# 15. EVIDENCE QUALITY VALIDATION
# --------------------------------------------------

def validate_evidence(claim, docs, llm):

    validated_docs = []

    validation_prompt = ChatPromptTemplate.from_template("""
You are an evidence quality evaluator.

Compare the document evidence with the user's claim.

CLAIM:
{claim}

DOCUMENT EVIDENCE:
{evidence}

Classify the evidence into exactly one category:

DIRECTLY RELEVANT
SUPPORTING CONTEXT
WEAKLY RELEVANT
IRRELEVANT

Rules:

DIRECTLY RELEVANT:
Directly provides information that supports or contradicts
the claim.

SUPPORTING CONTEXT:
Provides useful background for understanding the claim.

WEAKLY RELEVANT:
Mentions a related topic but does not provide useful
evidence for the claim.

IRRELEVANT:
Does not meaningfully relate to the claim.

Return only one category.
""")

    validation_chain = (
        validation_prompt
        | llm
        | StrOutputParser()
    )

    print("\nStarting evidence quality validation...")

    for index, doc in enumerate(docs, start=1):

        print(f"\nValidating evidence {index}...")

        result = validation_chain.invoke({
            "claim": claim,
            "evidence": doc.page_content
        })

        category = result.strip().upper()

        print("Quality category:", category)

        if "DIRECTLY RELEVANT" in category:
            validated_docs.append(doc)

        elif "SUPPORTING CONTEXT" in category:
            validated_docs.append(doc)

        elif "WEAKLY RELEVANT" in category:
            print("Evidence rejected: weak relevance.")

        elif "IRRELEVANT" in category:
            print("Evidence rejected: irrelevant.")

        else:
            print("Unrecognized response. Evidence rejected.")

    return validated_docs


# --------------------------------------------------
# 16. VALIDATE EVIDENCE
# --------------------------------------------------

if not top_docs:

    print("\nNo evidence passed the relevance filter.")
    print("Verdict: Insufficient Evidence")

    raise SystemExit


llm = create_llm()

try:

    validated_docs = validate_evidence(
        claim,
        top_docs,
        llm
    )

except Exception as error:

    raise RuntimeError(
        f"Evidence quality validation failed: {error}"
    ) from error


print("\nEvidence quality validation completed!")

print("Pages before validation:", len(top_docs))
print("Pages after validation:", len(validated_docs))


# --------------------------------------------------
# 17. STOP IF NO USEFUL EVIDENCE REMAINS
# --------------------------------------------------

if not validated_docs:

    print("\n" + "=" * 65)
    print("TRUTHLENS AI - EVIDENCE QUALITY RESULT")
    print("=" * 65)

    print("Claim:", claim)
    print("Verdict: Insufficient Evidence")

    print(
        "No directly relevant or supporting evidence "
        "passed quality validation."
    )

    raise SystemExit


# --------------------------------------------------
# 18. ASSIGN EVIDENCE IDS
# --------------------------------------------------

print("\nAssigning evidence IDs...")

evidence_records = []

for index, doc in enumerate(validated_docs, start=1):

    source = doc.metadata.get("source", "Unknown")
    page = doc.metadata.get("page", "Unknown")

    page_number = (
        page + 1 if isinstance(page, int) else page
    )

    evidence_records.append({
        "id": f"E{index}",
        "source": source,
        "page": page_number,
        "text": doc.page_content
    })

    print(
        f"[E{index}] {source}, Page {page_number}"
    )


# --------------------------------------------------
# 19. PREPARE EVIDENCE CONTEXT
# --------------------------------------------------

print("\nPreparing validated evidence context...")

context_parts = []

for evidence in evidence_records:

    context_parts.append(f"""
Evidence ID: [{evidence['id']}]

Source: {evidence['source']}

Page: {evidence['page']}

Evidence:
{evidence['text']}
""")

context = "\n\n".join(context_parts)

print("Validated evidence context prepared!")


# --------------------------------------------------
# 20. FACT-CHECKING PROMPT WITH CITATIONS
# --------------------------------------------------

prompt = ChatPromptTemplate.from_template("""
You are TruthLens AI, a document-based fact-checking assistant.

Analyze the user's claim using only the supplied evidence.

CLAIM:
{claim}

VALIDATED EVIDENCE:
{context}

IMPORTANT CITATION RULES:

1. Each evidence item has a unique ID such as [E1], [E2].
2. Cite the evidence ID after every important factual
   statement.
3. Use only evidence IDs provided in the context.
4. Never invent an evidence ID, document name or page number.
5. If a statement is supported by multiple evidence items,
   cite all relevant IDs.
6. Do not cite evidence that does not support the statement.
7. If there is insufficient evidence, say so clearly.

Choose exactly one verdict:

- Supported
- Contradicted
- Partially Supported
- Insufficient Evidence

Distinguish direct emissions from lifecycle emissions
when relevant.

Do not use outside knowledge.

REPORT FORMAT:

Claim:

Verdict:

Evidence and Explanation:

Source Documents:

Limitations:

Citations must use the format [E1], [E2], etc.
""")


# --------------------------------------------------
# 21. CREATE FACT-CHECKING CHAIN
# --------------------------------------------------

def create_chain(llm):

    return prompt | llm | StrOutputParser()


# --------------------------------------------------
# 22. GENERATE FINAL REPORT
# --------------------------------------------------

print("\nGenerating final fact-checking report...")

chain = create_chain(llm)

try:

    response = chain.invoke({
        "claim": claim,
        "context": context
    })

except Exception as error:

    raise RuntimeError(
        f"Groq report generation failed: {error}"
    ) from error


# --------------------------------------------------
# 23. CITATION VERIFICATION
# --------------------------------------------------

def verify_citations(response, evidence_records):

    print("\nStarting citation verification...")

    valid_ids = {
        evidence["id"]
        for evidence in evidence_records
    }

    cited_ids = set(
        re.findall(r"\[(E\d+)\]", response)
    )

    invalid_ids = cited_ids - valid_ids

    if invalid_ids:

        print("Invalid citation IDs found:", invalid_ids)

    else:

        print("All cited evidence IDs are valid.")

    if not cited_ids:

        print("WARNING: No evidence citations found.")

        return False, [], sorted(invalid_ids)

    if invalid_ids:

        return False, sorted(cited_ids), sorted(invalid_ids)

    return True, sorted(cited_ids), []


citation_success, cited_ids, invalid_ids = verify_citations(
    response,
    evidence_records
)


# --------------------------------------------------
# 24. DISPLAY FINAL REPORT
# --------------------------------------------------

print("\n" + "=" * 65)
print("          TRUTHLENS AI - V4.2")
print("=" * 65)

print("LLM Provider: Groq")

print("\n")
print(response)


# --------------------------------------------------
# 25. DISPLAY CITATION VERIFICATION RESULTS
# --------------------------------------------------

print("\n" + "=" * 65)
print("              CITATION VERIFICATION")
print("=" * 65)

if citation_success:

    print("Citation verification: PASSED")

else:

    print("Citation verification: WARNING")

    if not cited_ids:
        print("The report contains no evidence IDs.")

    if invalid_ids:
        print("Invalid evidence IDs:", invalid_ids)


print("\nEvidence citation reference:")

for evidence in evidence_records:

    print(
        f"[{evidence['id']}] "
        f"{evidence['source']} - Page {evidence['page']}"
    )


# --------------------------------------------------
# 26. DISPLAY RERANKING RESULTS
# --------------------------------------------------

print("\n" + "=" * 65)
print("              RERANKING RESULTS")
print("=" * 65)

print("PDF files:", len(pdf_files))
print("Pages:", len(documents))
print("Chunks after duplicate removal:", len(chunks))
print("Initially retrieved:", len(retrieved_docs))
print("Relevant chunks:", len(relevant_results))
print("Pages before quality validation:", len(top_docs))
print("Pages after quality validation:", len(validated_docs))


for index, (score, doc) in enumerate(
    ranked_results[:5],
    start=1
):

    source = doc.metadata.get("source", "Unknown")
    page = doc.metadata.get("page", "Unknown")

    page_number = (
        page + 1 if isinstance(page, int) else page
    )

    print(f"\nRank: {index}")
    print(f"Relevance score: {float(score):.4f}")
    print("Source:", source)
    print("Page:", page_number)


# --------------------------------------------------
# 27. DISPLAY VALIDATED EVIDENCE
# --------------------------------------------------

print("\n" + "=" * 65)
print("              VALIDATED EVIDENCE")
print("=" * 65)

for evidence in evidence_records:

    print(f"\nEvidence ID: [{evidence['id']}]")
    print("Source:", evidence["source"])
    print("Page:", evidence["page"])
    print("Text:", evidence["text"][:300], "...")


# --------------------------------------------------
# 28. FINAL STATUS
# --------------------------------------------------

print("\nLLM Provider Used: Groq")

print("\nV4.2 execution completed!")

print("Duplicate filtering completed!")
print("Cross-Encoder reranking completed!")
print("Relevance filtering completed!")
print("Source diversity completed!")
print("Evidence quality validation completed!")
print("Citation verification completed!")

print("TruthLens AI execution successful!")
