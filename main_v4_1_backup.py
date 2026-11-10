
import os
import hashlib
from pathlib import Path

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from sentence_transformers import CrossEncoder


# --------------------------------------------------
# 1. LOAD ENVIRONMENT VARIABLES
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

google_api_key = os.getenv("GOOGLE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

if not google_api_key and not groq_api_key:
    raise ValueError(
        "Both GOOGLE_API_KEY and GROQ_API_KEY are missing in .env"
    )


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

    duplicate_key = (
        source,
        page,
        content
    )

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
# 14. CREATE LLM PROVIDERS
# --------------------------------------------------

def create_llm():

    errors = []

    # Try Gemini

    if google_api_key:

        try:

            print("\nTrying Gemini...")

            llm = ChatGoogleGenerativeAI(
                model="gemini-3.8-flash",
                google_api_key=google_api_key,
                temperature=0
            )

            # Test provider with a small request

            llm.invoke("Reply with the word READY.")

            print("Gemini is available!")

            return llm, "Gemini"

        except Exception as error:

            print("Gemini unavailable:", error)

            errors.append(f"Gemini: {error}")

    # Try Groq

    if groq_api_key:

        try:

            print("\nSwitching to Groq...")

            llm = ChatGroq(
                model="openai/gpt-oss-20b",
                groq_api_key=groq_api_key,
                temperature=0,
                max_retries=1
            )

            llm.invoke("Reply with the word READY.")

            print("Groq is available!")

            return llm, "Groq (Backup)"

        except Exception as error:

            print("Groq unavailable:", error)

            errors.append(f"Groq: {error}")

    raise RuntimeError(
        "All LLM providers failed:\n" + "\n".join(errors)
    )


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

Important:
- Do not classify evidence as directly relevant merely
  because it mentions the same broad topic.
- Do not assume that related information proves the claim.
- Return only one category from the four categories above.
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

        # Keep only recognized useful categories

        if category == "DIRECTLY RELEVANT":

            validated_docs.append(doc)

        elif category == "SUPPORTING CONTEXT":

            validated_docs.append(doc)

        elif category == "WEAKLY RELEVANT":

            print("Evidence rejected: weak relevance.")

        elif category == "IRRELEVANT":

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


llm, provider = create_llm()

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
# 18. PREPARE VALIDATED EVIDENCE CONTEXT
# --------------------------------------------------

print("\nPreparing validated evidence context...")

context_parts = []

for doc in validated_docs:

    source = doc.metadata.get("source", "Unknown")

    page = doc.metadata.get("page", "Unknown")

    document_type = doc.metadata.get(
        "document_type",
        "Unknown"
    )

    page_number = (
        page + 1 if isinstance(page, int) else page
    )

    context_parts.append(f"""
Source: {source}

Page: {page_number}

Document Type: {document_type}

Evidence:
{doc.page_content}
""")

context = "\n\n".join(context_parts)

print("Validated evidence context prepared!")


# --------------------------------------------------
# 19. FACT-CHECKING PROMPT
# --------------------------------------------------

prompt = ChatPromptTemplate.from_template("""
You are TruthLens AI, a document-based fact-checking assistant.

Analyze the user's claim using only the validated evidence below.

CLAIM:
{claim}

VALIDATED EVIDENCE:
{context}

Instructions:

1. Choose exactly one verdict:

   - Supported
   - Contradicted
   - Partially Supported
   - Insufficient Evidence

2. Explain your reasoning clearly.

3. Mention the source document names and page numbers.

4. Do not invent information or references.

5. Identify evidence limitations.

6. Distinguish direct emissions from lifecycle emissions
   when relevant.

7. If evidence is insufficient, do not guess.

8. If evidence disagrees, explain the disagreement.

9. Do not claim that the documents prove more than
   the supplied evidence supports.

10. Use only the supplied evidence.

Report format:

Claim:

Verdict:

Evidence and Explanation:

Source Documents:

Limitations:
""")


# --------------------------------------------------
# 20. CREATE FACT-CHECKING CHAIN
# --------------------------------------------------

def create_chain(llm):

    return prompt | llm | StrOutputParser()


# --------------------------------------------------
# 21. GENERATE FINAL REPORT
# --------------------------------------------------

print("\nGenerating final fact-checking report...")

try:

    chain = create_chain(llm)

    response = chain.invoke({
        "claim": claim,
        "context": context
    })

except Exception as error:

    print("\nPrimary provider failed during report generation.")

    # Switch to Groq if Gemini was used

    if provider == "Gemini" and groq_api_key:

        print("Switching to Groq backup...")

        try:

            backup_llm = ChatGroq(
                model="openai/gpt-oss-20b",
                groq_api_key=groq_api_key,
                temperature=0,
                max_retries=1
            )

            backup_chain = create_chain(backup_llm)

            response = backup_chain.invoke({
                "claim": claim,
                "context": context
            })

            provider = "Groq (Backup)"

        except Exception as backup_error:

            raise RuntimeError(
                f"Gemini failed: {error}\n"
                f"Groq failed: {backup_error}"
            ) from backup_error

    else:

        raise RuntimeError(
            f"Report generation failed: {error}"
        ) from error


# --------------------------------------------------
# 22. DISPLAY FINAL REPORT
# --------------------------------------------------

print("\n" + "=" * 65)

print("          TRUTHLENS AI - V4.1")

print("=" * 65)

print("LLM Provider:", provider)

print("\n")

print(response)


# --------------------------------------------------
# 23. DISPLAY RERANKING RESULTS
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
# 24. DISPLAY VALIDATED EVIDENCE
# --------------------------------------------------

print("\n" + "=" * 65)

print("              VALIDATED EVIDENCE")

print("=" * 65)

for index, doc in enumerate(validated_docs, start=1):

    source = doc.metadata.get("source", "Unknown")

    page = doc.metadata.get("page", "Unknown")

    page_number = (
        page + 1 if isinstance(page, int) else page
    )

    print(f"\nEvidence {index}")

    print("Source:", source)

    print("Page:", page_number)

    print("Text:", doc.page_content[:300], "...")


# --------------------------------------------------
# 25. FINAL STATUS
# --------------------------------------------------

print("\nLLM Provider Used:", provider)

print("\nV4.1 execution completed!")

print("Duplicate filtering completed!")

print("Cross-Encoder reranking completed!")

print("Relevance filtering completed!")

print("Source diversity completed!")

print("Evidence quality validation completed!")

print("TruthLens AI execution successful!")
