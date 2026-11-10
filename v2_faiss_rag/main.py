
import os
from pathlib import Path

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

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
# 2. LOAD MULTIPLE PDF DOCUMENTS
# --------------------------------------------------

documents_path = BASE_DIR / "documents"

pdf_files = list(documents_path.glob("*.pdf"))

if not pdf_files:
    raise FileNotFoundError(
        "No PDF files found. Add PDFs to the documents folder."
    )

documents = []

print("\nLoading PDF documents...")

for pdf_file in pdf_files:

    print(f"Loading: {pdf_file.name}")

    loader = PyPDFLoader(str(pdf_file))

    pages = loader.load()

    documents.extend(pages)

print(f"\nTotal PDF files: {len(pdf_files)}")
print(f"Total pages loaded: {len(documents)}")


# --------------------------------------------------
# 3. SPLIT DOCUMENTS INTO CHUNKS
# --------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)

chunks = text_splitter.split_documents(documents)

print(f"Total chunks created: {len(chunks)}")


# --------------------------------------------------
# 4. LOAD LOCAL HUGGING FACE EMBEDDINGS
# --------------------------------------------------

print("\nLoading Hugging Face embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

print("Embedding model loaded successfully!")


# --------------------------------------------------
# 5. CREATE FAISS VECTOR DATABASE
# --------------------------------------------------

print("\nCreating FAISS vector store...")

vector_store = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)

print("FAISS vector store created successfully!")


# --------------------------------------------------
# 6. CREATE RETRIEVER
# --------------------------------------------------

retriever = vector_store.as_retriever(
    search_kwargs={"k": 4}
)


# --------------------------------------------------
# 7. INITIALIZE GEMINI LLM
# --------------------------------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    google_api_key=api_key,
    temperature=0
)


# --------------------------------------------------
# 8. CREATE PROMPT TEMPLATE
# --------------------------------------------------

prompt = ChatPromptTemplate.from_template("""
You are TruthLens AI, a document-based fact-checking assistant.

Analyze the user's claim using ONLY the retrieved evidence.

CLAIM:
{claim}

RETRIEVED EVIDENCE:
{context}

INSTRUCTIONS:

1. Classify the evidence as one of:
   - Supported
   - Contradicted
   - Partially Supported
   - Insufficient Evidence

2. Explain your reasoning clearly.
3. Use only the retrieved document evidence.
4. Do not invent facts or sources.
5. Mention document names and page numbers.
6. If evidence is insufficient, clearly state that.
7. Distinguish direct emissions from lifecycle emissions
   whenever relevant.
8. A contradiction in the documents does not automatically
   prove that a claim is false.

Generate the report using this format:

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

print("\nSearching for relevant evidence...")

retrieved_docs = retriever.invoke(claim)

if not retrieved_docs:
    print("No relevant evidence found.")
    raise SystemExit


# --------------------------------------------------
# 12. PREPARE CONTEXT WITH SOURCE INFORMATION
# --------------------------------------------------

context_parts = []

for doc in retrieved_docs:

    source = Path(
        doc.metadata.get("source", "Unknown")
    ).name

    page = doc.metadata.get("page")

    page_number = (
        page + 1 if isinstance(page, int) else "Unknown"
    )

    context_parts.append(
        f"""
Source: {source}
Page: {page_number}

Evidence:
{doc.page_content}
"""
    )

context = "\n\n".join(context_parts)


# --------------------------------------------------
# 13. GENERATE FACT-CHECKING REPORT
# --------------------------------------------------

print("\nAnalyzing claim using retrieved evidence...")

response = chain.invoke({
    "claim": claim,
    "context": context
})


# --------------------------------------------------
# 14. DISPLAY FINAL REPORT
# --------------------------------------------------

print("\n" + "=" * 65)
print("             TRUTHLENS AI - V2 FAISS RAG")
print("=" * 65)

print(response)

print("\n" + "=" * 65)
print("RETRIEVAL SUMMARY")
print("=" * 65)

print("PDF files loaded:", len(pdf_files))
print("Total pages:", len(documents))
print("Total chunks:", len(chunks))
print("Retrieved chunks:", len(retrieved_docs))

print("\nSource references:")

for doc in retrieved_docs:

    source = Path(
        doc.metadata.get("source", "Unknown")
    ).name

    page = doc.metadata.get("page")

    page_number = (
        page + 1 if isinstance(page, int) else "Unknown"
    )

    print(f"- {source} | Page {page_number}")

print("\nTruthLens AI analysis completed!")


