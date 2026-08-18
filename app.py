"""
AI Clinical Decision Support Lite
Streamlit interface for the Heart Failure / Hypertension guideline RAG system.
"""

import os
import re
import json

import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
import groq

# ============================================================
# PAGE CONFIG + STYLE
# ============================================================
st.set_page_config(
    page_title="AI Clinical Decision Support",
    page_icon="🫀",
    layout="centered",
)

st.markdown("""
<style>
    :root {
        --ink: #12242E;
        --paper: #F6F8F7;
        --line: #D8E0DD;
        --teal: #0E6E5C;
        --teal-dark: #0A4F42;
        --amber: #B8722A;
        --red: #A8362B;
    }
    html, body, [class*="css"] {
        font-family: "Source Sans Pro", "Segoe UI", sans-serif;
    }
    .stApp { background-color: var(--paper); }
    h1, h2, h3 { font-family: "Georgia", serif; color: var(--ink); }
    .app-header {
        border-bottom: 3px solid var(--teal);
        padding-bottom: 14px;
        margin-bottom: 6px;
    }
    .app-subtitle { color: #55645F; font-size: 0.95rem; }
    .source-badges span {
        display: inline-block;
        background: white;
        border: 1px solid var(--line);
        border-radius: 20px;
        padding: 4px 12px;
        margin-right: 8px;
        font-size: 0.78rem;
        color: var(--teal-dark);
    }
    .answer-card {
        background: white;
        border: 1px solid var(--line);
        border-left: 5px solid var(--teal);
        border-radius: 8px;
        padding: 22px 26px;
        margin-top: 18px;
    }
    .conf-High { color: var(--teal-dark); font-weight: 700; }
    .conf-Medium { color: var(--amber); font-weight: 700; }
    .conf-Low, .conf-Insufficient { color: var(--red); font-weight: 700; }
    .evidence-row {
        border-bottom: 1px solid var(--line);
        padding: 10px 0;
        font-size: 0.85rem;
    }
    .disclaimer-box {
        background: #FBF4EC;
        border: 1px solid #E8D5B7;
        border-radius: 8px;
        padding: 14px 18px;
        font-size: 0.82rem;
        color: #6B4A1E;
        margin-top: 24px;
    }
    .stTextInput input { font-size: 1.02rem; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================
st.markdown("""
<div class="app-header">
  <h1>🫀 AI Clinical Decision Support — Lite</h1>
  <div class="app-subtitle">Evidence-grounded answers from official clinical guidelines. Every recommendation is cited to its source, page, and section.</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="source-badges">
  <span>2021 ESC Heart Failure Guidelines</span>
  <span>WHO Hypertension Guideline (2021)</span>
</div>
""", unsafe_allow_html=True)

st.write("")

# ============================================================
# LOAD DATA + MODELS (cached so it only runs once)
# ============================================================
MIN_CONFIDENCE_SCORE = 0.30  # same threshold tuned in Day 2 miss analysis

SYSTEM_PROMPT = """You are an evidence-grounded clinical decision support assistant.

Rules you must always follow:
1. Use ONLY the retrieved guideline context provided below. Do not use outside medical knowledge.
2. If the retrieved context does not support an answer, say explicitly that the evidence is insufficient. Do not guess or fill gaps from general knowledge.
3. Do NOT provide patient-specific diagnosis or treatment. This includes any request that names or implies a specific real patient's characteristics (age, weight, renal function, comorbidities, current vitals, etc.) and asks for an exact dose, drug choice, or treatment decision for that patient. In these cases, do NOT calculate or state a specific dose — instead, state that this requires a clinician's direct assessment, and (if relevant) mention that the guideline provides general dosing ranges that a clinician can apply using their judgment.
4. Every recommendation you state must include a citation: document name, section, and page.
5. Prefer summarizing and comparing the retrieved chunks over quoting them at length. Short exact-wording excerpts are fine when precision of wording matters (e.g. a dose or threshold), but only when discussing the guideline's general recommendation — never when computing or confirming a dose for a specific patient scenario.
6. Detect patient-specific requests even when phrased indirectly (e.g. "my patient is...", "he weighs...", "what dose should I give him/her now"). Treat these the same as rule 3, regardless of how much clinical detail is provided.

Output format (always use this structure):
- Recommendation: short, direct, based only on retrieved chunks. If rule 3 applies, state clearly that a patient-specific dosing decision cannot be provided here.
- Supporting Evidence: bullet points mapped to the exact evidence used, with brief excerpts where useful.
- Citations: document name + section + page, one per bullet in Supporting Evidence.
- Confidence & Safety: one of High / Medium / Low / Insufficient Evidence, plus a one-line clinical disclaimer.
"""

PATIENT_SPECIFIC_REFUSAL = (
    "Recommendation: This question asks for a specific dosing or treatment decision "
    "for an individual patient based on their personal clinical characteristics. This "
    "system provides guideline-based information only and cannot make patient-specific "
    "treatment decisions.\n\n"
    "Supporting Evidence: Not applicable — this is a safety refusal, not an evidence gap.\n\n"
    "Citations: None.\n\n"
    "Confidence & Safety: Not Applicable. Patient-specific dosing must be determined by "
    "a treating clinician who has full access to the patient's history, labs, and "
    "clinical context. This system is a decision-support reference tool, not a replacement "
    "for clinical judgment."
)


@st.cache_resource(show_spinner="Loading embedding model...")
def load_model():
    return SentenceTransformer("all-mpnet-base-v2")


@st.cache_data(show_spinner="Loading guideline knowledge base...")
def load_data():
    with open("hf_esc_chunks_final.json", encoding="utf-8") as f:
        chunks = json.load(f)
    embeddings = np.load("hf_esc_chunk_embeddings.npy")
    return chunks, embeddings


def is_patient_specific_dosing_request(query: str) -> bool:
    query_lower = query.lower()
    patient_indicators = [
        r"\bmy patient\b", r"\bthis patient\b", r"\bthe patient is\b",
        r"\bhe is\b.*\byears old\b", r"\bshe is\b.*\byears old\b",
        r"\byears old\b.*\bweighs\b", r"\bweighs\b.*\bkg\b",
        r"\begfr\s*\d+", r"\bcreatinine\s*\d+",
    ]
    dosing_request_indicators = [
        r"\bexact dose\b", r"\bwhat dose should i give\b",
        r"\bhow much should i give\b", r"\bright now\b",
        r"\bwhat should i prescribe\b", r"\bcalculate.*dose\b",
    ]
    has_patient_context = any(re.search(p, query_lower) for p in patient_indicators)
    has_dosing_request = any(re.search(p, query_lower) for p in dosing_request_indicators)
    return has_patient_context and has_dosing_request


def semantic_search(query, model, chunks, embeddings, top_k=5):
    q_emb = model.encode([query], normalize_embeddings=True)
    sims = (embeddings @ q_emb.T).flatten()
    ranked = sims.argsort()[::-1][:top_k]
    return [(chunks[i], float(sims[i])) for i in ranked]


def is_evidence_sufficient(retrieved):
    return len(retrieved) > 0 and retrieved[0][1] >= MIN_CONFIDENCE_SCORE


def build_user_prompt(query, retrieved):
    context_blocks = []
    for i, (c, score) in enumerate(retrieved, start=1):
        citation = (
            f"{c['document_title']} — {c['section_title_en']}, "
            f"p.{c['page_start']}-{c['page_end']} "
            f"(chunk_id: {c['chunk_id']}, score: {score:.3f})"
        )
        context_blocks.append(f"[Chunk {i}] Citation: {citation}\n{c['text']}")
    context = "\n\n---\n\n".join(context_blocks)

    return f"""Clinical question: {query}

Retrieved guideline context (use ONLY this; cite the [Chunk N] source for every claim):

{context}

Answer using the required output format (Recommendation / Supporting Evidence / Citations / Confidence & Safety). If the context above does not actually answer the question, say so under Confidence & Safety as "Insufficient Evidence" rather than guessing."""


@st.cache_resource
def get_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY"))
    return groq.Groq(api_key=api_key)


def generate_answer(query, model, chunks, embeddings, top_k=None):
    top_k = top_k or 5  # FINAL_TOP_K from Day 2 (Precision@5 selection)
    if is_patient_specific_dosing_request(query):
        return {"query": query, "retrieved": [], "answer": PATIENT_SPECIFIC_REFUSAL}

    retrieved = semantic_search(query, model, chunks, embeddings, top_k=top_k)

    if not is_evidence_sufficient(retrieved):
        return {
            "query": query,
            "retrieved": retrieved,
            "answer": (
                "Recommendation: Insufficient evidence in the indexed guidelines to answer this "
                "question confidently.\n\nSupporting Evidence: None of the retrieved chunks scored "
                "above the confidence threshold.\n\nCitations: None.\n\nConfidence & Safety: "
                "Insufficient Evidence. This is not a substitute for clinical judgment or a full "
                "guideline review."
            ),
        }

    client = get_groq_client()
    user_prompt = build_user_prompt(query, retrieved)
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        max_tokens=800,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return {"query": query, "retrieved": retrieved, "answer": response.choices[0].message.content}


def confidence_class(answer_text):
    for level in ["Insufficient", "Low", "Medium", "High"]:
        if level in answer_text:
            return level
    return None


# ============================================================
# MAIN UI
# ============================================================
model = load_model()
chunks, embeddings = load_data()

example_qs = [
    "What is the first-line pharmacological treatment for HFrEF?",
    "What is the first-line treatment for hypertension in adults?",
    "What follow-up monitoring is recommended after starting an ACE inhibitor?",
]

query = st.text_input(
    "Ask a clinical question about heart failure or hypertension management",
    placeholder="e.g. What is the recommended diuretic use in heart failure?",
)

cols = st.columns(len(example_qs))
for col, eq in zip(cols, example_qs):
    if col.button(eq, use_container_width=True):
        query = eq

submitted = st.button("Get guideline-grounded answer", type="primary", use_container_width=True)

if (submitted or query) and query:
    with st.spinner("Retrieving evidence and generating answer..."):
        result = generate_answer(query, model, chunks, embeddings)

    conf = confidence_class(result["answer"])
    conf_html = f'<span class="conf-{conf}">{conf}</span>' if conf else ""

    st.markdown(f"""
    <div class="answer-card">
        {result["answer"].replace(chr(10), "<br>")}
    </div>
    """, unsafe_allow_html=True)

    if result["retrieved"]:
        with st.expander(f"📎 Evidence panel — {len(result['retrieved'])} retrieved chunks (for verification)"):
            for c, score in result["retrieved"]:
                st.markdown(f"""
                <div class="evidence-row">
                    <b>[{score:.3f}]</b> {c.get('document_title', c.get('document'))} —
                    {c.get('section_title_en', '')} (p.{c.get('page_start')}-{c.get('page_end')})
                    <br><span style="color:#666;">{c['text'][:300]}...</span>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer-box">
        ⚠️ This tool supports — never replaces — clinical judgment. It answers only from indexed
        official guidelines (ESC Heart Failure 2021, WHO Hypertension 2021) and refuses to
        provide patient-specific dosing or diagnosis.
    </div>
    """, unsafe_allow_html=True)
