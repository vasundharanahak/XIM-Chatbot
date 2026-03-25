import os
import glob
import pickle
import re
import platform
import numpy as np

from pathlib import Path
from pdf2image import convert_from_path
from bs4 import BeautifulSoup

import fitz
import pytesseract

from bert_score import score
from rouge_score import rouge_scorer
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from transformers.utils import logging
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings


logging.set_verbosity_error()



# ── PATHS ──
BASE_DIR = os.path.abspath(".")
DOCS_DIR = os.path.join(BASE_DIR, "university_docs")
VECTOR_STORE_PATH = os.path.join(BASE_DIR, "vectorstore.pkl")


# ── TESSERACT + POPPLER ──
system = platform.system()
if system == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Users\ylour\Desktop\thesis\Tesseract-OCR\tesseract.exe"
    POPPLER_PATH = r"C:\Users\ylour\Desktop\thesis\poppler-25.12.0\Library\bin"
elif system == "Darwin":
    pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"
    POPPLER_PATH = None
else:
    pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"
    POPPLER_PATH = None


# ── FASTAPI APP ──
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── IN-MEMORY CHAT HISTORY ──
chat_history = []
MAX_HISTORY_TURNS = 5  # keep last 5 Q&A pairs


# ── TEXT CLEANING ──
def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9.,;:!?()\n ]", "", text)
    return text.strip()


# ── PDF EXTRACTION ──
def extract_pdf(pdf_path):
    text = ""
    try:
        doc = fitz.open(pdf_path)
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
    except Exception as e:
        print(f"PDF read error: {e}")
        return ""
    if len(text.strip()) == 0:
        print(f"OCR running on {os.path.basename(pdf_path)}")
        try:
            if POPPLER_PATH:
                images = convert_from_path(pdf_path, poppler_path=POPPLER_PATH)
            else:
                images = convert_from_path(pdf_path)
            text = "\n".join(pytesseract.image_to_string(img) for img in images)
        except Exception as e:
            print(f"OCR error: {e}")
            return ""
    return clean_text(text)


# ── HTML EXTRACTION ──
def extract_html(raw_html):
    try:
        soup = BeautifulSoup(raw_html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator=" ")
        return clean_text(text)
    except Exception as e:
        print(f"HTML parse error: {e}")
        return ""


def format_history(history):
    if not history:
        return ""
    
    lines = []
    for turn in history:
        role = "User" if turn["role"] == "user" else "Xenia"
        lines.append(f"{role}: {turn['content']}")
    
    return "\n".join(lines)


def trim_history(history, max_turns):
    max_messages = max_turns * 2
    return history[-max_messages:] if len(history) > max_messages else history



# ── LOAD MODELS ──

print("Loading models...")
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

embedder = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5"
)

print("Models loaded.")

seen_files = set()

def add_file(file, raw_docs, text):
    name = os.path.basename(file)
    if name in seen_files:
        return
    seen_files.add(name)
    raw_docs.append(Document(page_content=text, metadata={"source": name}))


# ── LOAD / BUILD VECTORSTORE ──
def load_vectorstore():
    if os.path.exists(VECTOR_STORE_PATH):
        print("Loading existing vectorstore...")
        with open(VECTOR_STORE_PATH, "rb") as f:
            return pickle.load(f)

    print("Building vectorstore from documents...")
    raw_docs = []

    pdf_files  = glob.glob(f"{DOCS_DIR}/*.pdf")
    txt_files  = glob.glob(f"{DOCS_DIR}/*.txt")
    html_files = list(set(
        glob.glob(f"{DOCS_DIR}/*.html") +
        glob.glob(f"{DOCS_DIR}/*.htm")
    ))

    if len(pdf_files) == 0 and len(txt_files) == 0 and len(html_files) == 0:
        raise RuntimeError(f"No PDF, TXT or HTML files found in {DOCS_DIR}")

    for file in pdf_files:
        print(f"Processing {os.path.basename(file)}")
        text = extract_pdf(file)
        if text:
            add_file(file, raw_docs, text)
        else:
            print(f"  ⚠ Skipped (no text extracted): {os.path.basename(file)}")

    for file in txt_files:
        print(f"Processing {os.path.basename(file)}")
        text = Path(file).read_text(encoding="utf-8", errors="ignore")
        if text:
            add_file(file, raw_docs, text)
        else:
            print(f"  ⚠ Skipped (no text extracted): {os.path.basename(file)}")

    for file in html_files:
        print(f"Processing {os.path.basename(file)}")
        raw_html = Path(file).read_text(encoding="utf-8", errors="ignore")
        text = extract_html(raw_html)
        if text:
            add_file(file, raw_docs, text)
        else:
            print(f"  ⚠ Skipped (no text extracted): {os.path.basename(file)}")

    if len(raw_docs) == 0:
        raise RuntimeError("No content could be extracted from any document. Check Poppler/Tesseract installation.")

    print(f"Successfully extracted content from {len(raw_docs)} document(s).")

    splitter = RecursiveCharacterTextSplitter(chunk_size=800,chunk_overlap=150)
    docs = splitter.split_documents(raw_docs)
    print(f"Split into {len(docs)} chunks.")

    vectorstore = FAISS.from_documents(docs, embedder)

    with open(VECTOR_STORE_PATH, "wb") as f:
        pickle.dump(vectorstore, f)

    print("Vectorstore built and saved.")
    return vectorstore


vectorstore = load_vectorstore()

def evaluate_self_retrieval(vectorstore, k=8, sample_size=50):
    retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 8}
)

    all_docs = vectorstore.docstore._dict.values()
    all_docs = list(all_docs)

    if sample_size:
        all_docs = all_docs[:sample_size]

    total_precision = 0
    total_recall = 0
    total_hits = 0

    for doc in all_docs:
        source = doc.metadata.get("source")
        query = doc.page_content[:200]

        retrieved_docs = retriever.invoke(query)
        retrieved_sources = [d.metadata.get("source") for d in retrieved_docs]

        relevant_retrieved = retrieved_sources.count(source)

        precision = relevant_retrieved / k
        recall = 1 if relevant_retrieved > 0 else 0

        total_precision += precision
        total_recall += recall
        total_hits += 1 if relevant_retrieved > 0 else 0

    avg_precision = total_precision / len(all_docs)
    avg_recall = total_recall / len(all_docs)
    hit_rate = total_hits / len(all_docs)

    print("=== Self Retrieval Evaluation ===")
    print(f"Evaluated on {len(all_docs)} chunks")
    print(f"Precision@{k}: {avg_precision:.3f}")
    print(f"Recall@{k}: {avg_recall:.3f}")
    print(f"Hit@{k}: {hit_rate:.3f}")

evaluate_self_retrieval(vectorstore, k=4, sample_size=15)

def rerank_docs(query, docs):
    query_vec = embedder.embed_query(query)

    scored = []
    for doc in docs:
        doc_vec = embedder.embed_query(doc.page_content)
        score = np.dot(query_vec, doc_vec)
        scored.append((score, doc))

    scored.sort(reverse=True, key=lambda x: x[0])
    return [doc for _, doc in scored[:5]]


def compute_bertscore(reference, generated):
    """
    Compute BERTScore between reference and generated answer
    """

    P, R, F1 = score(
        [generated],
        [reference],
        lang="en",
        verbose=False
    )

    return F1.item()

evaluation_set = evaluation_set = [
    {
        "question": "What undergraduate and postgraduate programs are offered at XIM University?",
        "reference": "XIM University offers undergraduate, postgraduate, and doctoral programs across disciplines such as Business Management, Commerce, Computer Science, Engineering, Law, Economics, Liberal Arts, and Mass Communication."
    },
    {
        "question": "Where is XIM University located and how is the campus environment for students?",
        "reference": "XIM University is located in Harirajpur, Bhubaneswar, Odisha, India. The campus provides modern academic facilities, hostels, libraries, and spaces for extracurricular and student development activities."
    },
    {
        "question": "What is the admission process for MBA programs at XIM University?",
        "reference": "Admission to MBA programs at XIM University typically requires candidates to apply online and submit scores from entrance exams such as CAT, XAT, GMAT, or XAT. Shortlisted candidates may then go through group discussions and personal interviews."
    },
    {
        "question": "What are the hostel and accommodation facilities available for students?",
        "reference": "XIM University provides separate hostel facilities for male and female students with furnished rooms, dining halls, internet access, and recreational areas to support student life on campus."
    },
    {
        "question": "What scholarships or financial aid options are available for students?",
        "reference": "XIM University offers merit-based scholarships, need-based financial assistance, and special scholarships for academically outstanding students and those from economically weaker backgrounds."
    },
    {
        "question": "What are the placement opportunities for students graduating from XIM University?",
        "reference": "XIM University has a structured placement process where companies from sectors such as consulting, finance, IT, marketing, and analytics recruit students. The university provides placement support, internships, and career development programs."
    }
]

def evaluate_bertscore(vectorstore, evaluation_set):

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 8, "fetch_k": 20}
    )

    total_score = 0

    for item in evaluation_set:

        question = item["question"]
        reference = item["reference"]

        docs = retriever.invoke(question)

        context = "\n\n".join(
            d.page_content for d in docs
        )

        final_prompt = f"""Context from XIM University documents:
{context}

Question: {question}
"""

        generated_answer = chain.invoke({"input": final_prompt})

        bert = compute_bertscore(reference, generated_answer)

        total_score += bert

        print("=================================")
        print("Question:", question)
        print("Reference:", reference)
        print("Generated:", generated_answer)
        print("BERTScore:", round(bert,3))

    avg_score = total_score / len(evaluation_set)

    print("\n===== BERTScore Evaluation =====")
    print("Average BERTScore:", round(avg_score,3))



# ── PROMPT ──
# ── PROMPT ──
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are Xenia, the official AI assistant for XIM University.

RULES:
- Answer ONLY using the provided context.
- If the answer is partially available, answer as much as possible.
- If not found, say:
"I do not have that information right now — best to check with the university directly."

STYLE:
- Clear, natural, and helpful
- Use bullet points if useful
- Speak as "we"

IMPORTANT:
- Do NOT ignore relevant context
- Do NOT be overly restrictive
"""),
    ("human", "{input}")
])



# ── CHAIN ──
chain = prompt | llm | StrOutputParser()



# ── REQUEST MODEL ──
class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []


# ── ROUTES ──
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest):
    global chat_history

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 8, "fetch_k": 20}
    )
    
    docs = retriever.invoke(req.message)
    docs = rerank_docs(req.message, docs)
    
    context = "\n\n".join(
    d.page_content[:500] for d in docs
    )

    history_block = format_history(chat_history)

    if history_block:
        final_prompt = f"""Context:
{context}

Conversation so far:
{history_block}

User: {req.message}

Continue the conversation naturally.
"""
    else:
        final_prompt = f"""
        Context:
        {context}

        Question:
        {req.message}
        """

    try:
        answer = chain.invoke({"input": final_prompt})
    except Exception as e:
        print("LLM ERROR:", e)
        return {"answer": "Sorry, something went wrong. Try again."}

    # ✅ Update memory
    chat_history.append({"role": "user", "content": req.message})
    chat_history.append({"role": "assistant", "content": answer})

    # ✅ Trim memory
    chat_history = trim_history(chat_history, MAX_HISTORY_TURNS)

    return {"answer": answer}


evaluate_bertscore(vectorstore, evaluation_set)