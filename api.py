import os
import glob
import pickle
import re
import platform

from pathlib import Path
from pdf2image import convert_from_path

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


# ── NVIDIA API KEY ──
# Set this before running: set NVIDIA_API_KEY=your_key (Windows)
#                          export NVIDIA_API_KEY=your_key (Mac/Linux)
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", "")
if not NVIDIA_API_KEY:
    raise RuntimeError("NVIDIA_API_KEY environment variable not set!")


# ── LOAD MODELS ──
print("Loading models...")
llm = ChatNVIDIA(model="meta/llama-3.3-70b-instruct")
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
    pdf_files = glob.glob(f"{DOCS_DIR}/*.pdf")
    txt_files = glob.glob(f"{DOCS_DIR}/*.txt")

    for file in pdf_files:
        print(f"Processing {os.path.basename(file)}")
        text = extract_pdf(file)
        if text:
            raw_docs.append(Document(page_content=text, metadata={"source": file}))

    for file in txt_files:
        text = Path(file).read_text(encoding="utf-8", errors="ignore")
        raw_docs.append(Document(page_content=clean_text(text), metadata={"source": file}))

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.split_documents(raw_docs)
    vectorstore = FAISS.from_documents(docs, embedder)

    with open(VECTOR_STORE_PATH, "wb") as f:
        pickle.dump(vectorstore, f)

    print("Vectorstore built and saved.")
    return vectorstore

vectorstore = load_vectorstore()


# ── CHAIN ──
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer ONLY using the provided context."),
    ("human", "{input}"),
])
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
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(req.message)
    context = "\n\n".join(d.page_content for d in docs)
    final_prompt = f"Context:\n{context}\n\nQuestion:\n{req.message}\n"
    answer = chain.invoke({"input": final_prompt})
    return {"answer": answer}

