
import os
import re
import time
import pandas as pd

from pathlib import Path
from collections import Counter

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from sentence_transformers import CrossEncoder


# --------------------------------------------------
# 1. PATHS AND CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

PDF_PATH = BASE_DIR / "v2_faiss_rag" / "documents" / "GlobalEVOutlook2025.pdf"

CSV_PATH = BASE_DIR / "evaluation" / "test_questions.csv"

OUTPUT_PATH = BASE_DIR / "evaluation" / "retrieval_quality_results.csv"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


# --------------------------------------------------
# 2. LOAD TEST QUESTIONS
# --------------------------------------------------

print("\nLoading test questions...")

test_df = pd.read_csv(CSV_PATH)

print("Total questions:", len(test_df))


# --------------------------------------------------
# 3. LOAD PDF
# --------------------------------------------------

print("\nLoading PDF...")

loader = PyPDFLoader(str(PDF_PATH))
pages = loader.load()

print("Total pages:", len(pages))


# --------------------------------------------------
# 4. SPLIT DOCUMENTS INTO CHUNKS
# --------------------------------------------------

print("\nSplitting documents...")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

chunks = splitter.split_documents(pages)

print("Total chunks:", len(chunks))


# --------------------------------------------------
# 5. LOAD EMBEDDINGS
# --------------------------------------------------

print("\nLoading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)

print("Embedding model loaded!")


# --------------------------------------------------
# 6. BUILD FAISS FOR V2
# --------------------------------------------------

print("\nBuilding V2 FAISS...")

faiss_db = FAISS.from_documents(
    chunks,
    embeddings
)

print("V2 FAISS ready!")


# --------------------------------------------------
# 7. BUILD CHROMADB FOR V3 AND V4
# --------------------------------------------------

print("\nBuilding ChromaDB...")

chroma_db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="truthlens_retrieval_eval"
)

print("ChromaDB ready!")


# --------------------------------------------------
# 8. LOAD RERANKER FOR V4
# --------------------------------------------------

print("\nLoading reranker...")

reranker = CrossEncoder(RERANKER_MODEL)

print("Reranker ready!")


# --------------------------------------------------
# 9. HELPER FUNCTIONS
# --------------------------------------------------

def extract_expected_page(evidence):

    match = re.search(
        r"Page\s+(\d+)",
        str(evidence),
        re.IGNORECASE
    )

    if match:
        return int(match.group(1))

    return None


def tokenize(text):

    words = re.findall(
        r"[a-zA-Z0-9]+",
        str(text).lower()
    )

    stopwords = {
        "the", "a", "an", "is", "are", "was", "were",
        "in", "on", "of", "to", "and", "for", "by",
        "did", "does", "do", "that", "this", "with",
        "from", "over", "under", "more", "than"
    }

    return set(
        word for word in words
        if word not in stopwords and len(word) > 1
    )


def calculate_keyword_recall(expected, retrieved_text):

    expected_words = tokenize(expected)
    retrieved_words = tokenize(retrieved_text)

    if not expected_words:
        return 0.0

    matched_words = expected_words.intersection(retrieved_words)

    return len(matched_words) / len(expected_words)


def get_page_number(doc):

    # PyPDFLoader stores page numbers using zero-based indexing.
    # Convert to one-based page numbers for reporting.

    page = doc.metadata.get("page")

    if page is None:
        return None

    return int(page) + 1


# --------------------------------------------------
# 10. EVALUATE ALL THREE VERSIONS
# --------------------------------------------------

results = []

versions = [
    "V2_FAISS",
    "V3_CHROMADB",
    "V4_RERANKING"
]

for version in versions:

    print("\n" + "=" * 55)
    print("Evaluating:", version)
    print("=" * 55)

    for index, row in test_df.iterrows():

        question = str(row["question"])
        expected_evidence = str(row["expected_evidence"])

        expected_page = extract_expected_page(expected_evidence)

        print(
            f"\nQuestion {index + 1}: {question}"
        )

        start_time = time.time()

        # V2: FAISS retrieves top 4
        if version == "V2_FAISS":

            retrieved_docs = faiss_db.similarity_search(
                question,
                k=4
            )

        # V3: ChromaDB retrieves top 4
        elif version == "V3_CHROMADB":

            retrieved_docs = chroma_db.similarity_search(
                question,
                k=4
            )

        # V4: Retrieve top 20 and rerank to top 5
        else:

            candidates = chroma_db.similarity_search(
                question,
                k=20
            )

            pairs = [
                (question, doc.page_content)
                for doc in candidates
            ]

            scores = reranker.predict(pairs)

            ranked_docs = sorted(
                zip(candidates, scores),
                key=lambda item: item[1],
                reverse=True
            )

            retrieved_docs = [
                doc for doc, score in ranked_docs[:5]
            ]

        retrieval_time = time.time() - start_time

        # Combine retrieved text
        retrieved_text = "\n".join(
            doc.page_content for doc in retrieved_docs
        )

        # Extract retrieved page numbers
        retrieved_pages = [
            get_page_number(doc)
            for doc in retrieved_docs
        ]

        retrieved_pages = [
            page for page in retrieved_pages
            if page is not None
        ]

        # Check whether expected page was retrieved
        page_hit = (
            expected_page in retrieved_pages
            if expected_page is not None
            else None
        )

        # Calculate keyword overlap
        keyword_recall = calculate_keyword_recall(
            expected_evidence,
            retrieved_text
        )

        result = {
            "version": version,
            "question": question,
            "expected_evidence": expected_evidence,
            "expected_page": expected_page,
            "retrieved_pages": ", ".join(
                map(str, retrieved_pages)
            ),
            "page_hit": page_hit,
            "keyword_recall": round(keyword_recall, 4),
            "retrieval_time_seconds": round(
                retrieval_time, 4
            )
        }

        results.append(result)

        # Save after every question
        pd.DataFrame(results).to_csv(
            OUTPUT_PATH,
            index=False
        )

        print("Retrieved pages:", retrieved_pages)
        print("Expected page:", expected_page)
        print("Page hit:", page_hit)
        print("Keyword recall:", round(keyword_recall, 4))
        print("Retrieval time:", round(retrieval_time, 4))


# --------------------------------------------------
# 11. FINAL RESULTS
# --------------------------------------------------

results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("RETRIEVAL QUALITY EVALUATION COMPLETED")
print("=" * 60)

summary = results_df.groupby("version").agg(
    page_hit_rate=("page_hit", "mean"),
    average_keyword_recall=("keyword_recall", "mean"),
    average_retrieval_time=("retrieval_time_seconds", "mean")
).reset_index()

summary["page_hit_rate"] = (
    summary["page_hit_rate"] * 100
).round(2)

summary["average_keyword_recall"] = (
    summary["average_keyword_recall"] * 100
).round(2)

summary["average_retrieval_time"] = (
    summary["average_retrieval_time"]
).round(4)

print("\nFINAL COMPARISON:\n")

print(summary.to_string(index=False))

summary.to_csv(
    BASE_DIR / "evaluation" / "retrieval_quality_summary.csv",
    index=False
)

print("\nDetailed results saved to:")
print(OUTPUT_PATH)

print("\nSummary saved to:")
print(BASE_DIR / "evaluation" / "retrieval_quality_summary.csv")
