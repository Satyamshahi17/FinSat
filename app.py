import streamlit as st
from query_engine import answer_question
from sentiment_risk import run_pipeline

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="FinSat – Financial Intelligence System",
    page_icon="📊",
    layout="wide"
)

# ---------------- SESSION STATE ----------------
if "answer" not in st.session_state:
    st.session_state.answer = ""

# ---------------- STYLING ----------------
st.markdown("""
<style>
    .big-title {
        font-size: 36px;
        font-weight: 700;
    }
    .sub-title {
        font-size: 18px;
        color: #666;
    }
    .result-box {
        background-color: #f9f9f9;
        padding: 20px;
        border-radius: 10px;
        border-left: 6px solid #4CAF50;
        white-space: pre-wrap;
    }
    .risk-high { color: red; font-weight: bold; }
    .risk-medium { color: orange; font-weight: bold; }
    .risk-low { color: green; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("<div class='big-title'>📊 FinSat</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-title'>AI-powered Financial Q&A and Risk Analysis Platform</div>",
    unsafe_allow_html=True
)
st.divider()

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Mode Selection")

mode = st.sidebar.radio(
    "Choose Mode",
    ["Financial Q&A", "Sentiment & Risk Analysis"]
)

st.sidebar.info(
    "📌 This demo uses a **pre-indexed financial report of Infosys Limited for FY 2025-26 Q2**.\n\n"
    "Document ingestion & indexing are handled offline."
)

# ---------------- CACHE HEAVY FUNCTIONS ----------------
@st.cache_resource
def get_answer_fn():
    return answer_question

@st.cache_resource
def get_risk_fn():
    return run_pipeline

# ---------------- MAIN CONTENT ----------------
if mode == "Financial Q&A":
    st.subheader("💬 Financial Q&A")
    st.caption("Powered by Llama-3.1")
    st.caption("Try asking:")


    suggested_questions = [
        "Is the company facing liquidity risk?",
        "What does the report say about cash flow?",
        "What are the key risk factors mentioned?",
        "What is management’s outlook?",
        "Summarize the financial performance for Q2.",
        "Analyze the stock market data relating to shares of the company listed in India."
    ]

    for q in suggested_questions:
        st.caption(f"• {q}")

   

    query = st.text_input(
        "Ask a financial question",
        placeholder="e.g. What is the name of the company?"
    )

    if st.button("Get Answer") and query:
        with st.spinner("Analyzing report..."):
            answer_fn = get_answer_fn()
            st.session_state.answer = str(answer_fn(query))

    if st.session_state.answer:
        st.markdown("### ✅ Answer")
        st.write(st.session_state.answer)

elif mode == "Sentiment & Risk Analysis":
    st.subheader("FinBERT powered Sentiment & Risk Analytics")
    st.caption('Risk analysis is computed at the document level; section-level sentiment may differ.')

    if st.button("Run Risk Analysis"):
        with st.spinner("Running sentiment and risk analysis..."):
            risk_fn = get_risk_fn()
            report = risk_fn()

        risk = report["risk_summary"]

        st.markdown("### 📌 Risk Assessment")

        if risk["risk_level"] == "HIGH":
            st.markdown("<p class='risk-high'>HIGH RISK</p>", unsafe_allow_html=True)
        elif risk["risk_level"] == "MEDIUM":
            st.markdown("<p class='risk-medium'>MEDIUM RISK</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p class='risk-low'>LOW RISK</p>", unsafe_allow_html=True)

        st.write(risk.get("risk_reason", "N/A"))

        st.divider()
        st.markdown("### 🧠 Sentiment Highlights")

        # Display detailed category results
        if report.get("category_results"):
            st.divider()
            st.markdown("### 📊 Detailed Sentiment Results by Category")
            
            for category, results in report["category_results"].items():
                with st.expander(f"{category.upper()} ({len(results)} sentences)"):
                    for result in results:
                        sentiment = result['sentiment'].upper()
                        confidence = result['confidence']
                        sentence = result['sentence']
                        keywords = result.get('matched_keywords', [])
                        
                        st.markdown(
                            f"""
                            - **{sentiment}**  
                              _Confidence_: {confidence:.1%}  
                              _Keywords_: {', '.join(keywords)}  
                              _Text_: {sentence}
                            """
                        )
        
        if not report["risk_summary"]:
            st.info("No sentiment highlights detected.")

