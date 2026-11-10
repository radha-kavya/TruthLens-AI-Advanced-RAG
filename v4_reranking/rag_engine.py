
import os
import re
import tempfile
from pathlib import Path

import streamlit as st

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

from sentence_transformers import CrossEncoder


# ==================================================
# 1. PROJECT CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

DOCUMENTS_PATH = BASE_DIR / "documents"
CHROMA_PATH = BASE_DIR / "chroma_db"

COLLECTION_NAME = "truthlens_v4_clean"

MAX_EVIDENCE_PAGES = 5
MAX_PAGES_PER_SOURCE = 2

RELEVANCE_MARGIN = 3.0

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# ==================================================
# 2. LOAD GROQ API KEY
# ==================================================

def get_groq_api_key():

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:

        try:
            api_key = st.secrets["GROQ_API_KEY"]

        except Exception:
            api_key = None

    if not api_key:

        raise ValueError(
            "GROQ_API_KEY is missing. "
            "Configure it in Streamlit Cloud Secrets."
        )

    os.environ["GROQ_API_KEY"] = api_key

    return api_key


# ==================================================
# 3. LOAD PDF DOCUMENTS
# ==================================================

def load_pdf_documents(pdf_files):

    all_documents = []

    for pdf_file in pdf_files:

        loader = PyPDFLoader(str(pdf_file))

        pages = loader.load()

        for page in pages:

            page.metadata["source"] = pdf_file.name

            page.metadata["document_type"] = "PDF"

        all_documents.extend(pages)

    return all_documents


# ==================================================
# 4. CREATE CHROMADB
# ==================================================

def create_vector_database(pdf_files, embeddings):

    st.info(
        "Creating the document vector database. "
        "This may take several minutes on the first run."
    )

    # Load all PDF pages.
    documents = load_pdf_documents(pdf_files)

    if not documents:

        raise ValueError(
            "No content could be extracted from the PDF documents."
        )

    # Split documents into chunks.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    chunks = text_splitter.split_documents(documents)

    if not chunks:

        raise ValueError(
            "Document chunking produced no chunks."
        )

    # Streamlit Cloud application files may be read-only.
    # Therefore, use a writable temporary directory.
    deployment_path = (
        Path(tempfile.gettempdir())
        / "truthlens_chroma_db"
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(deployment_path)
    )

    return vector_store, len(chunks)


# ==================================================
# 5. LOAD OR CREATE CHROMADB
# ==================================================

def load_or_create_vector_database(pdf_files, embeddings):

    vector_store = None

    # Try loading an existing local database.
    if CHROMA_PATH.exists():

        try:

            existing_store = Chroma(
                collection_name=COLLECTION_NAME,
                persist_directory=str(CHROMA_PATH),
                embedding_function=embeddings
            )

            stored_count = existing_store._collection.count()

            if stored_count > 0:

                vector_store = existing_store

        except Exception:

            vector_store = None

    # Create a fresh database if missing or empty.
    if vector_store is None:

        vector_store, chunk_count = create_vector_database(
            pdf_files,
            embeddings
        )

        st.success(
            f"ChromaDB created successfully with {chunk_count} chunks."
        )

    stored_chunks = vector_store._collection.count()

    if stored_chunks == 0:

        raise ValueError(
            "ChromaDB collection is empty."
        )

    return vector_store, stored_chunks


# ==================================================
# 6. LOAD ALL RAG RESOURCES
# ==================================================

@st.cache_resource(show_spinner="Loading TruthLens AI models...")
def load_resources():

    # ----------------------------------------------
    # GROQ API KEY
    # ----------------------------------------------

    groq_api_key = get_groq_api_key()

    # ----------------------------------------------
    # FIND PDF DOCUMENTS
    # ----------------------------------------------

    pdf_files = sorted(
        DOCUMENTS_PATH.glob("*.pdf")
    )

    if not pdf_files:

        raise FileNotFoundError(
            "No PDF documents found inside the documents folder."
        )

    if len(pdf_files) < 3:

        raise FileNotFoundError(
            f"Expected 3 PDF documents, but found {len(pdf_files)}."
        )

    # ----------------------------------------------
    # LOAD EMBEDDING MODEL
    # ----------------------------------------------

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        }
    )

    # ----------------------------------------------
    # LOAD OR CREATE VECTOR DATABASE
    # ----------------------------------------------

    vector_store, stored_chunks = load_or_create_vector_database(
        pdf_files,
        embeddings
    )

    # ----------------------------------------------
    # CREATE RETRIEVER
    # ----------------------------------------------

    retriever = vector_store.as_retriever(
        search_kwargs={
            "k": 20
        }
    )

    # ----------------------------------------------
    # LOAD CROSS-ENCODER
    # ----------------------------------------------

    reranker = CrossEncoder(
        RERANKER_MODEL,
        device="cpu"
    )

    # ----------------------------------------------
    # LOAD GROQ LLM
    # ----------------------------------------------

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        groq_api_key=groq_api_key,
        temperature=0,
        max_retries=1
    )

    return {
        "pdf_files": pdf_files,
        "vector_store": vector_store,
        "retriever": retriever,
        "reranker": reranker,
        "llm": llm,
        "stored_chunks": stored_chunks
    }


# ==================================================
# 7. EVIDENCE QUALITY VALIDATION
# ==================================================

def validate_evidence(claim, docs, llm):

    validated_docs = []

    directly_relevant_docs = []

    quality_counts = {
        "DIRECTLY RELEVANT": 0,
        "SUPPORTING CONTEXT": 0,
        "WEAKLY RELEVANT": 0,
        "IRRELEVANT": 0,
        "UNRECOGNIZED": 0
    }

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

    categories = [
        "DIRECTLY RELEVANT",
        "SUPPORTING CONTEXT",
        "WEAKLY RELEVANT",
        "IRRELEVANT"
    ]

    for doc in docs:

        result = validation_chain.invoke({
            "claim": claim,
            "evidence": doc.page_content
        })

        result_text = result.strip().upper()

        category = next(
            (
                item for item in categories
                if re.search(
                    rf"\b{re.escape(item)}\b",
                    result_text
                )
            ),
            "UNRECOGNIZED"
        )

        quality_counts[category] += 1

        doc.metadata["evidence_quality"] = category

        if category == "DIRECTLY RELEVANT":

            directly_relevant_docs.append(doc)

            validated_docs.append(doc)

        elif category == "SUPPORTING CONTEXT":

            validated_docs.append(doc)

    return (
        validated_docs,
        quality_counts,
        directly_relevant_docs
    )


# ==================================================
# 8. ASSIGN EVIDENCE IDS
# ==================================================

def create_evidence_records(validated_docs):

    evidence_records = []

    for index, doc in enumerate(
        validated_docs,
        start=1
    ):

        source = doc.metadata.get(
            "source",
            "Unknown"
        )

        page = doc.metadata.get(
            "page",
            "Unknown"
        )

        page_number = (
            page + 1
            if isinstance(page, int)
            else page
        )

        evidence_records.append({

            "id": f"E{index}",

            "source": source,

            "page": page_number,

            "category": doc.metadata.get(
                "evidence_quality",
                "UNRECOGNIZED"
            ),

            "text": doc.page_content

        })

    return evidence_records


# ==================================================
# 9. PREPARE EVIDENCE CONTEXT
# ==================================================

def prepare_evidence_context(evidence_records):

    context_parts = []

    for evidence in evidence_records:

        context_parts.append(f"""
Evidence ID: [{evidence['id']}]

Source: {evidence['source']}

Page: {evidence['page']}

Evidence:
{evidence['text']}
""")

    return "\n\n".join(context_parts)


# ==================================================
# 10. FACT-CHECKING PROMPT
# ==================================================

prompt = ChatPromptTemplate.from_template("""
You are TruthLens AI, a document-based fact-checking assistant.

Analyze the user's claim using only the supplied evidence.

CLAIM:
{claim}

VALIDATED EVIDENCE:
{context}

IMPORTANT CITATION RULES:

1. Each evidence item has a unique ID such as [E1], [E2].
2. Cite evidence IDs after important factual statements.
3. Use only evidence IDs provided in the context.
4. Never invent an evidence ID, document name or page number.
5. Cite all relevant evidence items when appropriate.
6. Do not cite evidence that does not support the statement.
7. If there is insufficient evidence, say so clearly.
8. Always write evidence IDs using square brackets.

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


# ==================================================
# 11. CITATION VERIFICATION
# ==================================================

def verify_citations(response, evidence_records):

    valid_ids = {
        evidence["id"]
        for evidence in evidence_records
    }

    cited_ids = set(
        re.findall(
            r"(?<![A-Za-z0-9])\[?(E\d+)\]?(?![A-Za-z0-9])",
            response,
            re.IGNORECASE
        )
    )

    cited_ids = {
        item.upper()
        for item in cited_ids
    }

    invalid_ids = cited_ids - valid_ids

    if not cited_ids:

        return False, [], []

    return (
        len(invalid_ids) == 0,
        sorted(cited_ids),
        sorted(invalid_ids)
    )


# ==================================================
# 12. EXTRACT VERDICT
# ==================================================

def extract_verdict(response):

    pattern = (
        r"Verdict\s*:\s*"
        r"(Supported|Contradicted|Partially Supported|"
        r"Inconclusive|Insufficient Evidence)"
    )

    match = re.search(
        pattern,
        response,
        re.IGNORECASE
    )

    if match:

        return match.group(1).title()

    return "Verdict not identified"


# ==================================================
# 13. EMPTY RESULT HELPER
# ==================================================

def create_empty_result(
    claim,
    report,
    pdf_count,
    stored_chunks,
    retrieved_chunks=0,
    relevant_chunks=0,
    evidence_pages=0,
    validated_pages=0,
    quality_counts=None
):

    return {
        "claim": claim,
        "verdict": "Insufficient Evidence",
        "confidence": "Low",
        "report": report,
        "citation_success": False,
        "cited_ids": [],
        "invalid_ids": [],
        "evidence_records": [],
        "quality_counts": quality_counts or {},
        "ranked_results": [],
        "stats": {
            "pdf_files": pdf_count,
            "retrieved_chunks": retrieved_chunks,
            "relevant_chunks": relevant_chunks,
            "evidence_pages": evidence_pages,
            "validated_pages": validated_pages,
            "stored_chunks": stored_chunks
        }
    }


# ==================================================
# 14. MAIN CLAIM ANALYSIS FUNCTION
# ==================================================

def analyze_claim(claim):

    claim = claim.strip()

    if not claim:

        raise ValueError(
            "Please enter a valid claim."
        )

    # ----------------------------------------------
    # LOAD RESOURCES
    # ----------------------------------------------

    resources = load_resources()

    retriever = resources["retriever"]

    reranker = resources["reranker"]

    llm = resources["llm"]

    pdf_files = resources["pdf_files"]

    stored_chunks = resources["stored_chunks"]

    # ----------------------------------------------
    # RETRIEVAL
    # ----------------------------------------------

    retrieved_docs = retriever.invoke(claim)

    if not retrieved_docs:

        return create_empty_result(
            claim,
            "No relevant documents were retrieved.",
            len(pdf_files),
            stored_chunks
        )

    # ----------------------------------------------
    # CROSS-ENCODER RERANKING
    # ----------------------------------------------

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

    # ----------------------------------------------
    # RELEVANCE FILTERING
    # ----------------------------------------------

    highest_score = max(
        float(score)
        for score, doc in ranked_results
    )

    relevance_threshold = (
        highest_score - RELEVANCE_MARGIN
    )

    relevant_results = [
        (score, doc)
        for score, doc in ranked_results
        if float(score) >= relevance_threshold
    ]

    # ----------------------------------------------
    # SOURCE DIVERSITY
    # ----------------------------------------------

    top_docs = []

    seen_pages = set()

    source_counts = {}

    # First pass:
    # Maximum two pages per PDF.

    for score, doc in relevant_results:

        source = doc.metadata.get(
            "source",
            "Unknown"
        )

        page = doc.metadata.get(
            "page",
            "Unknown"
        )

        page_key = (source, page)

        if page_key in seen_pages:
            continue

        if source_counts.get(source, 0) >= MAX_PAGES_PER_SOURCE:
            continue

        seen_pages.add(page_key)

        top_docs.append(doc)

        source_counts[source] = (
            source_counts.get(source, 0) + 1
        )

        if len(top_docs) == MAX_EVIDENCE_PAGES:
            break

    # Second pass:
    # Fill remaining evidence slots.

    if len(top_docs) < MAX_EVIDENCE_PAGES:

        for score, doc in relevant_results:

            source = doc.metadata.get(
                "source",
                "Unknown"
            )

            page = doc.metadata.get(
                "page",
                "Unknown"
            )

            page_key = (source, page)

            if page_key in seen_pages:
                continue

            seen_pages.add(page_key)

            top_docs.append(doc)

            source_counts[source] = (
                source_counts.get(source, 0) + 1
            )

            if len(top_docs) == MAX_EVIDENCE_PAGES:
                break

    # ----------------------------------------------
    # NO EVIDENCE CHECK
    # ----------------------------------------------

    if not top_docs:

        return create_empty_result(
            claim,
            "No evidence passed the relevance filter.",
            len(pdf_files),
            stored_chunks,
            len(retrieved_docs),
            len(relevant_results)
        )

    # ----------------------------------------------
    # EVIDENCE QUALITY VALIDATION
    # ----------------------------------------------

    (
        validated_docs,
        quality_counts,
        directly_relevant_docs
    ) = validate_evidence(
        claim,
        top_docs,
        llm
    )

    # ----------------------------------------------
    # INSUFFICIENT EVIDENCE CHECK
    # ----------------------------------------------

    if quality_counts["DIRECTLY RELEVANT"] == 0:

        return create_empty_result(
            claim,
            (
                "No directly relevant evidence passed quality "
                "validation. Supporting context alone cannot "
                "establish whether the claim is true or false."
            ),
            len(pdf_files),
            stored_chunks,
            len(retrieved_docs),
            len(relevant_results),
            len(top_docs),
            len(validated_docs),
            quality_counts
        )

    # ----------------------------------------------
    # ASSIGN EVIDENCE IDS
    # ----------------------------------------------

    evidence_records = create_evidence_records(
        validated_docs
    )

    # ----------------------------------------------
    # PREPARE CONTEXT
    # ----------------------------------------------

    context = prepare_evidence_context(
        evidence_records
    )

    # ----------------------------------------------
    # GENERATE FINAL REPORT
    # ----------------------------------------------

    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({

        "claim": claim,

        "context": context

    })

    # ----------------------------------------------
    # CITATION VERIFICATION
    # ----------------------------------------------

    (
        citation_success,
        cited_ids,
        invalid_ids
    ) = verify_citations(
        response,
        evidence_records
    )

    # ----------------------------------------------
    # QUALITATIVE CONFIDENCE
    # ----------------------------------------------

    direct_count = quality_counts[
        "DIRECTLY RELEVANT"
    ]

    direct_source_count = len({

        doc.metadata.get(
            "source",
            "Unknown"
        )

        for doc in directly_relevant_docs

    })

    if (
        citation_success
        and direct_count >= 2
        and direct_source_count >= 2
    ):

        confidence_level = "High"

    elif citation_success and direct_count >= 1:

        confidence_level = "Medium"

    else:

        confidence_level = "Low"

    # ----------------------------------------------
    # PREPARE RERANKING RESULTS
    # ----------------------------------------------

    ranking_records = []

    for index, (score, doc) in enumerate(
        ranked_results[:5],
        start=1
    ):

        page = doc.metadata.get(
            "page",
            "Unknown"
        )

        page_number = (
            page + 1
            if isinstance(page, int)
            else page
        )

        ranking_records.append({

            "rank": index,

            "score": round(
                float(score),
                4
            ),

            "source": doc.metadata.get(
                "source",
                "Unknown"
            ),

            "page": page_number

        })

    # ----------------------------------------------
    # RETURN RESULTS TO STREAMLIT
    # ----------------------------------------------

    return {

        "claim": claim,

        "verdict": extract_verdict(response),

        "confidence": confidence_level,

        "report": response,

        "citation_success": citation_success,

        "cited_ids": cited_ids,

        "invalid_ids": invalid_ids,

        "evidence_records": evidence_records,

        "quality_counts": quality_counts,

        "ranked_results": ranking_records,

        "stats": {

            "pdf_files": len(pdf_files),

            "retrieved_chunks": len(retrieved_docs),

            "relevant_chunks": len(relevant_results),

            "evidence_pages": len(top_docs),

            "validated_pages": len(validated_docs),

            "stored_chunks": stored_chunks

        }

    }
