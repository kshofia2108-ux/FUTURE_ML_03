import io
import re
from typing import List
import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# nltk.download('punkt'); nltk.download('stopwords')
STOPWORDS = set(nltk.corpus.stopwords.words("english"))
MAX_BYTES = 200 * 1024 * 1024  # 200 MB

def is_pdf_file(f: st.runtime.uploaded_file_manager.UploadedFile) -> bool:
    # check filename extension and MIME type if available
    name_ok = f.name.lower().endswith(".pdf")
    mime_ok = getattr(f, "type", "") in ("application/pdf", "application/x-pdf", "")
    return name_ok and mime_ok

def extract_text_from_pdf_bytes(b: bytes) -> str:
    text_parts = []
    with io.BytesIO(b) as f:
        reader = PdfReader(f)
        for p in reader.pages:
            t = p.extract_text()
            if t:
                text_parts.append(t)
    return " ".join(text_parts or [""])

def clean_text(text: str) -> str:
    t = (text or "").lower()
    t = re.sub(r"[\r\n]+", " ", t)
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    tokens = [w for w in nltk.word_tokenize(t) if w not in STOPWORDS and len(w) > 1]
    return " ".join(tokens)

def semantic_scores(resume_texts: List[str], job_text: str) -> List[float]:
    vec = TfidfVectorizer(max_features=5000)
    docs = resume_texts + [job_text]
    tfidf = vec.fit_transform(docs)
    job_vec = tfidf[-1]
    sims = cosine_similarity(tfidf[:-1], job_vec).flatten()
    return sims.tolist()

st.set_page_config(page_title="📄 Resume Screening (PDF only)", layout="wide")
st.title("📄 Resume Screening (PDF only)")

st.markdown("Upload PDF files only. Each file must be a PDF and <= 200 MB.")

job_file = st.file_uploader("📋 Upload Job Description (PDF, up to 200 MB)", type=["pdf"], accept_multiple_files=False)
resume_files = st.file_uploader("📑 Upload Resume(s) (PDF, multiple files, up to 200 MB)", type=["pdf"], accept_multiple_files=True)

if job_file:
    if job_file.size > MAX_BYTES:
        st.error("Job description exceeds 200 MB limit.")
        job_file = None
    elif not is_pdf_file(job_file):
        st.error("Job description must be a valid PDF file.")
        job_file = None

valid_resumes = []
if resume_files:
    for f in resume_files:
        if f.size > MAX_BYTES:
            st.error(f"Resume '{f.name}' exceeds 200 MB limit and was skipped.")
            continue
        if not is_pdf_file(f):
            st.error(f"Resume '{f.name}' is not a valid PDF and was skipped.")
            continue
        valid_resumes.append(f)

if st.button("▶️ Analyze"):
    if job_file is None or not valid_resumes:
        st.error("Please upload a valid job description PDF and at least one valid resume PDF.")
    else:
        try:
            job_raw = extract_text_from_pdf_bytes(job_file.read())
            job_clean = clean_text(job_raw)
        except Exception as e:
            st.error(f"Failed to parse job PDF: {e}")
            st.stop()

        resume_texts = []
        resume_names = []
        for rf in valid_resumes:
            try:
                txt = extract_text_from_pdf_bytes(rf.read())
                resume_texts.append(clean_text(txt))
                resume_names.append(rf.name)
            except Exception as e:
                st.warning(f"Failed to parse '{rf.name}': {e}")

        if not resume_texts:
            st.error("No resumes could be parsed.")
            st.stop()

        # simple keyword extraction from job (top words)
        words = [w for w in job_clean.split() if len(w) > 2]
        keywords = pd.Series(words).value_counts().index[:12].tolist()

        sem_scores = semantic_scores(resume_texts, job_clean)
        kw_scores = []
        for txt in resume_texts:
            wset = set(txt.split())
            kw_scores.append(sum(1 for k in keywords if k in wset) / max(1, len(keywords)))

        combined = []
        for n, s, k in zip(resume_names, sem_scores, kw_scores):
            score = 0.65 * s + 0.35 * k
            combined.append({"name": n, "combined": score, "semantic": float(s), "keywords": float(k)})

        df = pd.DataFrame(combined).sort_values("combined", ascending=False).reset_index(drop=True)
        df["rank"] = df.index + 1

        st.subheader("📊 Candidate Ranking")
        st.dataframe(df[["rank", "name", "combined", "semantic", "keywords"]].round(3))

        best = df.iloc[0]
        st.subheader("🏆 Best Candidate")
        st.write(f"Name: **{best['name']}** — Combined: **{best['combined']:.3f}**")

        csv_bytes = df[["rank", "name", "combined", "semantic", "keywords"]].to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download CSV", data=csv_bytes, file_name="ranking.csv", mime="text/csv")
