
import os
from pathlib import Path

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# --------------------------------------------------
# 1. LOAD ENVIRONMENT VARIABLES
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file")


# --------------------------------------------------
# 2. LOAD PDF DOCUMENTS
# --------------------------------------------------

documents_path = BASE_DIR / "documents"

pdf_files = list(documents_path.glob("*.pdf"))

if not pdf_files:
    raise FileNotFoundError("No PDF files found in documents folder.")

documents = []

print("\nLoading PDF documents...")

for pdf_file in pdf_files:

    print("Loading:", pdf_file.name)

    loader = PyPDFLoader(str(pdf_file))

    pages = loader.load()

    # Add metadata to every page
    for page in pages:
        page.metadata["source"] = pdf_file.name
        page.metadata["document_type"] = "Research Report"

    documents.extend(pages)

print("Total PDFs:", len(pdf_files))
print("Total pages:", len(documents))


# --------------------------------------------------
# 3. SPLIT DOCUMENTS INTO CHUNKS
# --------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)

chunks = text_splitter.split_documents(documents)

print("Total chunks:", len(chunks))


# --------------------------------------------------
# 4. LOAD LOCAL EMBEDDINGS
# --------------------------------------------------

print("\nLoading Hugging Face embeddings...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

print("Embeddings loaded successfully!")


# --------------------------------------------------
# 5. CREATE CHROMADB VECTOR DATABASE
# --------------------------------------------------

print("\nCreating ChromaDB vector database...")

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="truthlens_documents",
    persist_directory=str(BASE_DIR / "chroma_db")
)

print("ChromaDB created successfully!")


# --------------------------------------------------
# 6. CREATE RETRIEVER
# --------------------------------------------------

retriever = vector_store.as_retriever(
    search_kwargs={"k": 4}
)


# --------------------------------------------------
# 7. INITIALIZE GEMINI
# --------------------------------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    google_api_key=api_key,
    temperature=0
)


# --------------------------------------------------
# 8. CREATE PROMPT
# --------------------------------------------------

prompt = ChatPromptTemplate.from_template("""
You are TruthLens AI, a document-based fact-checking assistant.

Analyze the claim using only the retrieved evidence.

CLAIM:
{claim}

RETRIEVED EVIDENCE:
{context}

Instructions:

1. Give one verdict:
   - Supported
   - Contradicted
   - Partially Supported
   - Insufficient Evidence

2. Explain the evidence clearly.
3. Mention source documents and page numbers.
4. Do not invent facts.
5. Identify limitations in the available evidence.
6. Distinguish direct emissions from lifecycle emissions
   whenever relevant.

Use this report format:

Claim:
Verdict:
Evidence and Explanation:
Source Documents:
Limitations:
""")


# --------------------------------------------------
# 9. CREATE LANGCHAIN CHAIN
# --------------------------------------------------

chain = prompt | llm | StrOutputParser()


# --------------------------------------------------
# 10. GET USER CLAIM
# --------------------------------------------------

claim = input("\nEnter a claim to analyze: ").strip()

if not claim:
    raise ValueError("Please enter a valid claim.")


# --------------------------------------------------
# 11. RETRIEVE RELEVANT DOCUMENTS
# --------------------------------------------------

print("\nSearching ChromaDB...")

retrieved_docs = retriever.invoke(claim)

if not retrieved_docs:
    print("No relevant evidence found.")
    raise SystemExit


# --------------------------------------------------
# 12. PREPARE CONTEXT
# --------------------------------------------------

context_parts = []

for doc in retrieved_docs:

    source = doc.metadata.get("source", "Unknown")
    page = doc.metadata.get("page", "Unknown")
    document_type = doc.metadata.get(
        "document_type", "Unknown"
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


# --------------------------------------------------
# 13. GENERATE REPORT
# --------------------------------------------------

print("\nAnalyzing evidence...")

response = chain.invoke({
    "claim": claim,
    "context": context
})


# --------------------------------------------------
# 14. DISPLAY REPORT
# --------------------------------------------------

print("\n" + "=" * 65)
print("          TRUTHLENS AI - V3 CHROMADB RAG")
print("=" * 65)

print(response)


# --------------------------------------------------
# 15. DISPLAY METADATA
# --------------------------------------------------

print("\n" + "=" * 65)
print("              RETRIEVAL METADATA")
print("=" * 65)

print("PDF files:", len(pdf_files))
print("Pages:", len(documents))
print("Chunks:", len(chunks))
print("Retrieved chunks:", len(retrieved_docs))

for index, doc in enumerate(retrieved_docs, start=1):

    source = doc.metadata.get("source", "Unknown")
    page = doc.metadata.get("page", "Unknown")
    document_type = doc.metadata.get(
        "document_type", "Unknown"
    )

    page_number = (
        page + 1 if isinstance(page, int) else page
    )

    print(f"\nEvidence {index}")
    print("Source:", source)
    print("Page:", page_number)
    print("Document Type:", document_type)

print("\nV3 ChromaDB analysis completed!")
