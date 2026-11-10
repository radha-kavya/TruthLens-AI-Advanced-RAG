
import re
import html
from pathlib import Path

import streamlit as st
import pandas as pd

from rag_engine import analyze_claim


# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="TruthLens AI",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ==================================================
# 2. MAROON AND ORANGE THEME
# ==================================================

st.markdown("""
<style>

.stApp {
    background-color: #2B1018;
    color: #FFFFFF;
}

.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

/* Headings */

h1, h2, h3, h4 {
    color: #FFFFFF !important;
}

/* General text */

p, label, li {
    color: #F5E9EC;
}

/* Text area */

.stTextArea textarea {
    background-color: #4A202D !important;
    color: #FFFFFF !important;
    border: 1px solid #80505D !important;
    border-radius: 10px;
}

.stTextArea textarea::placeholder {
    color: #D8BBC3 !important;
}

/* Buttons */

.stButton > button {
    background-color: #FF963F !important;
    color: #2B1018 !important;
    border: none !important;
    border-radius: 9px;
    font-weight: bold;
    padding: 0.65rem 1.5rem;
}

.stButton > button:hover {
    background-color: #FFB16F !important;
    color: #2B1018 !important;
}

/* Metric cards */

.metric-card {
    background-color: #4A202D;
    border: 1px solid #75414F;
    border-radius: 12px;
    padding: 18px;
    text-align: center;
    min-height: 105px;
}

.metric-label {
    color: #E0C8CE;
    font-size: 14px;
    margin-bottom: 8px;
}

.metric-value {
    color: #FF963F;
    font-size: 23px;
    font-weight: bold;
    overflow-wrap: anywhere;
}

/* Expanders */

[data-testid="stExpander"] {
    background-color: #3A1722 !important;
    border: 1px solid #75414F !important;
    border-radius: 10px;
}

[data-testid="stExpander"] summary {
    color: #FFFFFF !important;
}

/* Evidence content */

.evidence-content {
    background-color: #4A202D;
    border-left: 4px solid #FF963F;
    border-radius: 7px;
    padding: 14px;
    color: #FFFFFF;
    line-height: 1.6;
    overflow-wrap: anywhere;
}

/* Dividers */

hr {
    border-color: #75414F !important;
}

/* Links */

a {
    color: #FF963F !important;
}

/* Dataframe */

[data-testid="stDataFrame"] {
    background-color: #3A1722;
}

/* Chart text */

[data-testid="stVegaLiteChart"] text {
    fill: #FFFFFF !important;
}

</style>
""", unsafe_allow_html=True)


# ==================================================
# 3. HELPER FUNCTIONS
# ==================================================

def display_metric(label, value):

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">
                {html.escape(str(label))}
            </div>
            <div class="metric-value">
                {html.escape(str(value))}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ==================================================
# 4. EXTRACT VERDICT
# ==================================================

def extract_verdict(report, result):

    verdict = result.get("verdict")

    allowed_verdicts = [
        "Supported",
        "Contradicted",
        "Partially Supported",
        "Inconclusive",
        "Insufficient Evidence"
    ]

    if verdict in allowed_verdicts:
        return verdict

    if not report:
        return "Not identified"

    pattern = (
        r"\*{0,2}Verdict\*{0,2}\s*:\s*"
        r"\*{0,2}\s*"
        r"(Supported|Contradicted|Partially Supported|"
        r"Inconclusive|Insufficient Evidence)"
    )

    match = re.search(
        pattern,
        report,
        re.IGNORECASE
    )

    if match:
        return match.group(1).title()

    return "Not identified"


# ==================================================
# 5. DISPLAY RETRIEVED EVIDENCE
# ==================================================

def display_evidence(evidence_records):

    st.subheader("Retrieved Evidence")

    if not evidence_records:
        st.info("No evidence records available.")
        return

    for index, evidence in enumerate(evidence_records, start=1):

        if not isinstance(evidence, dict):
            continue

        evidence_id = evidence.get(
            "id",
            evidence.get("evidence_id", f"E{index}")
        )

        source = evidence.get(
            "source",
            "Unknown source"
        )

        page = evidence.get(
            "page",
            "Not specified"
        )

        category = evidence.get(
            "category",
            "UNRECOGNIZED"
        )

        content = evidence.get(
            "text",
            evidence.get(
                "content",
                evidence.get("page_content", "")
            )
        )

        source_name = Path(str(source)).name

        with st.expander(
            f"{evidence_id} | {source_name} | Page {page}"
        ):

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Evidence ID**")
                st.write(evidence_id)

            with col2:
                st.write("**Evidence Quality**")
                st.write(str(category).title())

            st.write("**Source Document**")
            st.write(source_name)

            st.write("**Page**")
            st.write(page)

            st.write("**Evidence Content**")

            if content:

                st.markdown(
                    f"""
                    <div class="evidence-content">
                        {html.escape(str(content))}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:
                st.info("Evidence text is not available.")


# ==================================================
# 6. EVIDENCE QUALITY DISTRIBUTION
# ==================================================

def display_quality_chart(quality_counts):

    st.subheader("Evidence Quality Distribution")

    categories = [
        "DIRECTLY RELEVANT",
        "SUPPORTING CONTEXT",
        "WEAKLY RELEVANT",
        "IRRELEVANT"
    ]

    if not isinstance(quality_counts, dict) or not quality_counts:
        st.info("Evidence quality statistics are not available.")
        return

    chart_rows = []

    for category in categories:

        raw_count = quality_counts.get(category, 0)

        try:
            count = int(raw_count)
        except (TypeError, ValueError):
            count = 0

        chart_rows.append({
            "Evidence Quality": category.title(),
            "Count": max(0, count)
        })

    chart_data = pd.DataFrame(chart_rows)

    total = int(chart_data["Count"].sum())

    if total == 0:
        st.info("No classified evidence items to display.")
        return

    st.bar_chart(
        chart_data,
        x="Evidence Quality",
        y="Count",
        horizontal=True
    )

    st.caption(
        f"Total classified evidence items: {total}"
    )


# ==================================================
# 7. CITATION VERIFICATION DISPLAY
# ==================================================

def display_citation_verification(
    cited_ids,
    invalid_ids,
    citation_status
):

    st.subheader("Citation Verification")

    col1, col2 = st.columns(2)

    with col1:

        st.write("**Cited Evidence IDs**")

        if cited_ids:
            st.write(", ".join(map(str, cited_ids)))
        else:
            st.write("No cited IDs available.")

    with col2:

        st.write("**Invalid Evidence IDs**")

        if invalid_ids:
            st.write(", ".join(map(str, invalid_ids)))
        else:
            st.write("None")

    if citation_status is True:

        st.success(
            "Citation verification successful."
        )

    elif citation_status is False:

        st.warning(
            "Some citations could not be verified."
        )

    else:

        st.info(
            "Citation verification status is unavailable."
        )


# ==================================================
# 8. MAIN APPLICATION
# ==================================================

st.title("TruthLens AI")

st.write(
    "Analyze claims using evidence retrieved from multiple "
    "documents. Review the supporting evidence and citations "
    "behind the generated report."
)

st.divider()

st.subheader("Enter a Claim")

claim = st.text_area(
    "What would you like to fact-check?",
    placeholder=(
        "Example: Electric vehicles always produce zero emissions."
    ),
    height=120,
    key="claim_input"
)

analyze_button = st.button(
    "🔍 Analyze Claim",
    use_container_width=True
)


# ==================================================
# 9. CLAIM ANALYSIS
# ==================================================

if analyze_button:

    # Clear old results before starting a new analysis.
    st.session_state.pop("truthlens_result", None)
    st.session_state.pop("truthlens_claim", None)

    if not claim.strip():

        st.warning(
            "Please enter a claim before starting the analysis."
        )

    else:

        try:

            with st.spinner(
                "Retrieving documents, reranking evidence "
                "and analyzing the claim..."
            ):

                result = analyze_claim(claim.strip())

            if not isinstance(result, dict):

                st.error(
                    "The RAG engine returned an unexpected result."
                )

            else:

                st.session_state["truthlens_result"] = result

                st.session_state["truthlens_claim"] = claim.strip()

        except Exception:

            st.error(
                "An error occurred during claim analysis. "
                "Please check your model connection and application logs."
            )

            st.exception(
                __import__("traceback").format_exc()
            )


# ==================================================
# 10. DISPLAY ANALYSIS RESULTS
# ==================================================

if "truthlens_result" in st.session_state:

    result = st.session_state["truthlens_result"]

    report = result.get(
        "report",
        result.get(
            "response",
            result.get("answer", "")
        )
    )

    confidence = result.get(
        "confidence",
        "Not available"
    )

    citation_status = result.get(
        "citation_success",
        None
    )

    evidence_records = result.get(
        "evidence_records",
        []
    )

    quality_counts = result.get(
        "quality_counts",
        {}
    )

    cited_ids = result.get(
        "cited_ids",
        []
    )

    invalid_ids = result.get(
        "invalid_ids",
        []
    )

    verdict = extract_verdict(report, result)

    st.divider()

    st.header("Analysis Results")

    # ----------------------------------------------
    # METRICS
    # ----------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:
        display_metric(
            "VERDICT",
            verdict
        )

    with col2:
        display_metric(
            "CONFIDENCE",
            confidence
        )

    with col3:

        if citation_status is True:
            citation_label = "Verified"

        elif citation_status is False:
            citation_label = "Failed"

        else:
            citation_label = "Not available"

        display_metric(
            "CITATION STATUS",
            citation_label
        )

    st.divider()

    # ----------------------------------------------
    # FACT-CHECKING REPORT
    # ----------------------------------------------

    st.header("Fact-Checking Report")

    if report:
        st.markdown(report)

    else:
        st.warning(
            "The RAG engine did not return a report."
        )

    st.divider()

    # ----------------------------------------------
    # RETRIEVED EVIDENCE
    # ----------------------------------------------

    display_evidence(evidence_records)

    st.divider()

    # ----------------------------------------------
    # EVIDENCE QUALITY DISTRIBUTION
    # ----------------------------------------------

    display_quality_chart(quality_counts)

    st.divider()

    # ----------------------------------------------
    # CITATION VERIFICATION
    # ----------------------------------------------

    display_citation_verification(
        cited_ids,
        invalid_ids,
        citation_status
    )
