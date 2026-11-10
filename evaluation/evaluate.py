
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import CrossEncoder


# ---------------- CONFIGURATION ----------------

ROOT = Path(__file__).resolve().parents[1]

DOCUMENTS_DIR = ROOT / "v2_faiss_rag" / "documents"
QUESTIONS_FILE = ROOT / "evaluation" / "test_questions.csv"
OUTPUT_FILE = ROOT / "evaluation" / "evaluation_results.csv"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GEMINI_MODEL = "gemini-3.8-flash"

TOP_K = 4
RETRIEVE_K = 20
RERANK_TOP_K = 5

# Only one Gemini call per version
LLM_QUESTIONS_PER_VERSION = 1


# ---------------- LOAD API KEY ----------------

load_dotenv(ROOT / "v2_faiss_rag" / ".env")

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError(
        "GOOGLE_API_KEY not found. Check v2_faiss_rag/.env"
    )


# ---------------- LOAD QUESTIONS ----------------

questions_df = pd.read_csv(QUESTIONS_FILE)

print(f"Total questions: {len(questions_df)}")


# ---------------- LOAD PDF DOCUMENTS ----------------

print("\nLoading PDF documents...")

documents = []

for pdf_file in DOCUMENTS_DIR.glob("*.pdf"):
    print(f"Loading: {pdf_file.name}")

    loader = PyPDFLoader(str(pdf_file))
    pdf_docs = loader.load()

    for doc in pdf_docs:
        doc.metadata["source"] = pdf_file.name
        doc.metadata["document_type"] = "Research Report"

    documents.extend(pdf_docs)

if not documents:
    raise FileNotFoundError(
        f"No PDF files found in {DOCUMENTS_DIR}"
    )

print(f"Total pages: {len(documents)}")


# ---------------- SPLIT DOCUMENTS ----------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)

chunks = splitter.split_documents(documents)

print(f"Total chunks: {len(chunks)}")


# ---------------- LOAD EMBEDDINGS ----------------

print("\nLoading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)

print("Embeddings loaded!")


# ---------------- BUILD V2 FAISS ----------------

print("\nBuilding V2 FAISS...")

faiss_db = FAISS.from_documents(
    chunks,
    embeddings
)

v2_retriever = faiss_db.as_retriever(
    search_kwargs={"k": TOP_K}
)

print("V2 ready!")


# ---------------- BUILD V3 CHROMA ----------------

print("\nBuilding V3 ChromaDB...")

v3_db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="truthlens_eval_v3"
)

v3_retriever = v3_db.as_retriever(
    search_kwargs={"k": TOP_K}
)

print("V3 ready!")


# ---------------- BUILD V4 CHROMA ----------------

print("\nBuilding V4 ChromaDB...")

v4_db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="truthlens_eval_v4"
)

v4_retriever = v4_db.as_retriever(
    search_kwargs={"k": RETRIEVE_K}
)

print("V4 ready!")


# ---------------- LOAD RERANKER ----------------

print("\nLoading Cross-Encoder...")

reranker = CrossEncoder(RERANKER_MODEL)

print("Reranker loaded!")


# ---------------- LOAD GEMINI ----------------

print("\nInitializing Gemini...")

llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    temperature=0,
    max_output_tokens=250
)

prompt = ChatPromptTemplate.from_template("""
You are a document-based fact-checking assistant.

Analyze the claim using ONLY the supplied evidence.

Claim: {question}

Evidence:
{context}

Return a short response in this exact format:

Verdict: Supported / Contradicted / Partially Supported / Insufficient Evidence
Explanation: One or two short sentences.
""")

print("Gemini initialized!")


# ---------------- HELPER FUNCTIONS ----------------

def get_page(doc):
    page = doc.metadata.get("page")

    if page is not None:
        return int(page) + 1

    return "Unknown"


def prepare_context(docs):
    parts = []

    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        page = get_page(doc)

        parts.append(
            f"Source: {source}\n"
            f"Page: {page}\n"
            f"Evidence: {doc.page_content[:1200]}"
        )

    return "\n\n".join(parts)


def extract_verdict(answer):
    valid_verdicts = [
        "Partially Supported",
        "Insufficient Evidence",
        "Contradicted",
        "Supported"
    ]

    for verdict in valid_verdicts:
        if f"Verdict: {verdict}".lower() in answer.lower():
            return verdict

    return "Unclear"


def save_results(results):
    pd.DataFrame(results).to_csv(
        OUTPUT_FILE,
        index=False
    )


# ---------------- EVALUATION ----------------

results = []

versions = [
    ("V2_FAISS", v2_retriever),
    ("V3_CHROMADB", v3_retriever),
    ("V4_RERANKING", v4_retriever)
]

print("\nStarting retrieval evaluation...")


for version_name, retriever in versions:

    print("\n" + "=" * 60)
    print(f"Evaluating {version_name}")
    print("=" * 60)

    for index, row in questions_df.iterrows():

        question = str(row["question"])
        expected = str(row["expected_verdict"])

        print(f"\nQuestion {index + 1}: {question}")

        start_time = time.time()

        # Retrieve documents
        retrieved_docs = retriever.invoke(question)

        # V4 reranking
        if version_name == "V4_RERANKING":

            pairs = [
                (question, doc.page_content)
                for doc in retrieved_docs
            ]

            scores = reranker.predict(pairs)

            ranked_docs = sorted(
                zip(scores, retrieved_docs),
                key=lambda item: item[0],
                reverse=True
            )

            retrieved_docs = [
                doc for score, doc in ranked_docs[:RERANK_TOP_K]
            ]

        retrieval_time = time.time() - start_time

        sources = "; ".join(
            f"{doc.metadata.get('source', 'Unknown')} "
            f"(Page {get_page(doc)})"
            for doc in retrieved_docs
        )

        context = prepare_context(retrieved_docs)

        # Generate Gemini answer only for first question
        generation_time = 0
        predicted = "Not Evaluated"
        verdict_matched = "Not Evaluated"
        error = ""

        if index < LLM_QUESTIONS_PER_VERSION:

            print("Generating short Gemini answer...")

            try:
                generation_start = time.time()

                response = llm.invoke(
                    prompt.format(
                        question=question,
                        context=context
                    )
                )

                generation_time = (
                    time.time() - generation_start
                )

                answer = response.content

                if isinstance(answer, list):
                    answer = " ".join(
                        str(item) for item in answer
                    )

                answer = str(answer)

                predicted = extract_verdict(answer)
                verdict_matched = (
                    predicted.lower() == expected.lower()
                )

                print(f"Expected: {expected}")
                print(f"Predicted: {predicted}")
                print(f"Matched: {verdict_matched}")

            except Exception as e:
                error = str(e)
                predicted = "ERROR"

                print(f"Gemini error: {error}")

        else:
            print("Retrieval completed; Gemini call skipped.")

        print(f"Retrieval time: {retrieval_time:.2f} seconds")

        if generation_time:
            print(
                f"Generation time: {generation_time:.2f} seconds"
            )

        results.append({
            "version": version_name,
            "question": question,
            "expected_verdict": expected,
            "predicted_verdict": predicted,
            "verdict_matched": verdict_matched,
            "retrieval_time_seconds": round(retrieval_time, 3),
            "generation_time_seconds": round(generation_time, 3),
            "retrieved_sources": sources,
            "error": error
        })

        # Save after every question
        save_results(results)


# ---------------- FINAL SUMMARY ----------------

print("\n" + "=" * 60)
print("EVALUATION COMPLETED")
print("=" * 60)

results_df = pd.DataFrame(results)

print("\nRetrieval performance:")

print(
    results_df.groupby("version")[
        "retrieval_time_seconds"
    ].mean().round(3)
)

print("\nGemini evaluation results:")

llm_results = results_df[
    results_df["predicted_verdict"] != "Not Evaluated"
]

print(
    llm_results[
        [
            "version",
            "expected_verdict",
            "predicted_verdict",
            "verdict_matched"
        ]
    ].to_string(index=False)
)

print(f"\nResults saved to: {OUTPUT_FILE}")
