import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(env_path, override=True)

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY is missing from the .env file.")

print("Starting TruthLens AI...", flush=True)

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    google_api_key=api_key,
    temperature=0
)

prompt = ChatPromptTemplate.from_template("""
You are TruthLens AI, a preliminary fact-checking assistant.

Analyze this claim: {claim}

No documents or external evidence have been provided.
Do not claim that the statement has been verified.

Use these headings:
1. Claim
2. Preliminary Assessment
3. Evidence Required
4. Limitations

Clearly state that this assessment is preliminary.
""")

chain = prompt | llm | StrOutputParser()

if __name__ == "__main__":
    claim = input("Enter a claim to analyze: ").strip()

    if not claim:
        print("Please enter a claim.")
    else:
        try:
            print("\nAnalyzing claim. Please wait...", flush=True)
            result = chain.invoke({"claim": claim})
            print("\n--- TruthLens AI Report ---\n")
            print(result)
        except Exception as error:
            print("\nUnable to generate the report.")
            print(f"Error type: {type(error).__name__}")
            print(f"Error details: {error}")

