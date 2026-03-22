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
    model="llama-3.3-70b-versatile",   # best balance of speed + quality
    temperature=0
)

embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Models loaded.")


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
    html_files = glob.glob(f"{DOCS_DIR}/*.html") + glob.glob(f"{DOCS_DIR}/*.htm")

    if len(pdf_files) == 0 and len(txt_files) == 0 and len(html_files) == 0:
        raise RuntimeError(f"No PDF, TXT or HTML files found in {DOCS_DIR}")

    for file in pdf_files:
        print(f"Processing {os.path.basename(file)}")
        text = extract_pdf(file)
        if text:
            raw_docs.append(Document(page_content=text, metadata={"source": os.path.basename(file)}))
        else:
            print(f"  ⚠ Skipped (no text extracted): {os.path.basename(file)}")

    for file in txt_files:
        print(f"Processing {os.path.basename(file)}")
        text = Path(file).read_text(encoding="utf-8", errors="ignore")
        if text:
            raw_docs.append(Document(page_content=clean_text(text), metadata={"source": os.path.basename(file)}))
        else:
            print(f"  ⚠ Skipped (no text extracted): {os.path.basename(file)}")

    for file in html_files:
        print(f"Processing {os.path.basename(file)}")
        raw_html = Path(file).read_text(encoding="utf-8", errors="ignore")
        text = extract_html(raw_html)
        if text:
            raw_docs.append(Document(page_content=text, metadata={"source": os.path.basename(file)}))
        else:
            print(f"  ⚠ Skipped (no text extracted): {os.path.basename(file)}")

    if len(raw_docs) == 0:
        raise RuntimeError("No content could be extracted from any document. Check Poppler/Tesseract installation.")

    print(f"Successfully extracted content from {len(raw_docs)} document(s).")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.split_documents(raw_docs)
    print(f"Split into {len(docs)} chunks.")

    vectorstore = FAISS.from_documents(docs, embedder)

    with open(VECTOR_STORE_PATH, "wb") as f:
        pickle.dump(vectorstore, f)

    print("Vectorstore built and saved.")
    return vectorstore

vectorstore = load_vectorstore()



# ── PROMPT ──
# ── PROMPT ──
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are Xenia, the official AI assistant for XIM University. You're helpful, sharp, and to the point — like a well-informed friend who works at the university.

    Rules:
    - Be concise but not robotic. When asked about a subject, include all details you are sure about.
    - Be human and conversational. Write like a person, not a policy document.
    - Structure answers when helpful. Use bullet points for multiple items or lists.
    - Be accurate. Only state facts you are confident about.
    - Answer as if you already know the information internally.
    - Never reveal or hint at where the information came from.

    Strict constraints:
    - NEVER mention "context", "documents", "provided information", "knowledge base", or anything similar.
    - NEVER say phrases like:
    "according to the context"
    "based on the provided information"
    "the document states"
    "the context does not mention"
    - NEVER explain how you know the answer.

    Missing information rule:
    - If you are not certain about an answer or the information is not available, say:
    "I do not have that information right now — best to check with the university directly."
    - Do NOT speculate or invent details.
    - Do NOT explain that the information is missing from documents.

    Tone and identity:
    - Always refer to XIM University as "we", "our", or "us".
    - Never say "they" or "XIM University does/offers".
    - Do not fabricate names, numbers, dates, designations, or contact details.

    Response style:
    - Match the length to the question.
    - Short question → short answer.
    - Complex query → structured but concise answer.

    Avoid filler openings such as:
    "Great question!"
    "Certainly!"
    "Of course!"
    """),
    ("human", "{input}"),
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

    context = "\n\n".join(
        f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
        for d in docs
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
        final_prompt = f"""Context:
{context}

Question: {req.message}
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