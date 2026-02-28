import os
import glob
import pickle
import re
import platform

from pathlib import Path
from pdf2image import convert_from_path
from bs4 import BeautifulSoup

import fitz
import pytesseract

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


# ── PATHS ──
BASE_DIR = os.path.abspath(".")
DOCS_DIR = os.path.join(BASE_DIR, "university_docs")
VECTOR_STORE_PATH = os.path.join(BASE_DIR, "vectorstore.pkl")


# ── TESSERACT + POPPLER ──
system = platform.system()
if system == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    POPPLER_PATH = r"C:\poppler\Library\bin"
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


# ── NVIDIA API KEY ──
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")
if not NVIDIA_API_KEY:
    raise RuntimeError("NVIDIA_API_KEY environment variable not set!")


# ── LOAD MODELS ──
print("Loading models...")
llm = ChatNVIDIA(model="meta/llama-3.3-70b-instruct", temperature=0)
embedder = NVIDIAEmbeddings(model="nvidia/nv-embedqa-e5-v5", model_type="passage")
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
    ("system", """You are Envie, the official AI assistant for XIM University. You're helpful, sharp, and to the point — like a well-informed friend who works at the university.

Rules:
- Be concise but don't be robotic. Be concise but when asked about a subject, try to provide all details you are aware of. Answer politely cheerfully.
- Be human. Write like a person, not a policy document.
- Be structured when it helps. Use bullet points for when there are multiple distinct items to convey. But when asked for lists, do use bullet points. 
- Be accurate. Only state facts that are clearly present in the documents provided. If something isn't there, say "I don't have that info — best to check with XIM directly!" and leave it at that. Don't mention anything about the "provided context".
- Never fabricate names, numbers, dates, designations, or contact details. Ever.
- Don't start answers with "Great question!", "Certainly!", "Of course!" or any filler phrase.
- Don't reference the documents explicitly — just answer as if you already know this.
- Always use "we", "our", "us" when referring to XIM University. Never "they" or "XIM University does/offers".
- Match the length to the question. One-line question? One or two line answer. Complex query? Give a structured but tight response.
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
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 8, "fetch_k": 20})
    
    # Expand query slightly for better retrieval
    expanded_query = f"{req.message} XIM University"
    docs = retriever.invoke(req.message)
    context = "\n\n".join(
        f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
        for d in docs
    )
    final_prompt = f"""Context from XIM University documents:
{context}

Question: {req.message}
"""
    answer = chain.invoke({"input": final_prompt})
    return {"answer": answer}