import os
import glob
import pickle
import re
import platform
import streamlit as st
import fitz
import pytesseract
import pandas as pd

from pathlib import Path
from pdf2image import convert_from_path

from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Envie Chatbot",
    layout="wide"
)

st.title("🤖 Envie — NVIDIA-Powered RAG Chatbot")
st.write("Chat with your university documents")


# =====================================================
# PATH CONFIG
# =====================================================

BASE_DIR = os.path.abspath(".")
DOCS_DIR = os.path.join(BASE_DIR, "university_docs")
VECTOR_STORE_PATH = os.path.join(BASE_DIR, "vectorstore.pkl")


# =====================================================
# TESSERACT + POPPLER SETUP (Windows)
# =====================================================

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


# =====================================================
# TEXT CLEANING
# =====================================================

def clean_text(text):

    text = re.sub(r"\s+", " ", text)

    text = re.sub(
        r"[^a-zA-Z0-9.,;:!?()\n ]",
        "",
        text
    )

    return text.strip()


# =====================================================
# PDF EXTRACTION WITH OCR
# =====================================================

def extract_pdf(pdf_path):
    text = ""
    try:
        doc = fitz.open(pdf_path)
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
    except Exception as e:
        print(f"PDF read error: {e}")
        return ""

    # Only use OCR if text is extremely short
    if len(text.strip()) < 50:
        print(f"OCR running on {os.path.basename(pdf_path)}")
        try:
            images = convert_from_path(pdf_path, poppler_path=POPPLER_PATH)
            text = "\n".join(pytesseract.image_to_string(img) for img in images)
        except Exception as e:
            print(f"OCR error: {e}")
            return ""

    return clean_text(text)
# =====================================================
# NVIDIA API KEY
# =====================================================

if "NVIDIA_API_KEY" not in os.environ:

    api_key = st.text_input(
        "Enter NVIDIA API Key",
        type="password"
    )

    if api_key:

        os.environ["NVIDIA_API_KEY"] = api_key

    else:

        st.stop()


# =====================================================
# INITIALIZE MODELS
# =====================================================

@st.cache_resource
def initialize_models():

    llm = ChatNVIDIA(
        model="meta/llama-3.3-70b-instruct"
    )

    embedder = NVIDIAEmbeddings(
        model="nvidia/nv-embedqa-e5-v5",
        model_type="passage"
    )

    return llm, embedder


llm, embedder = initialize_models()


# =====================================================
# LOAD OR CREATE VECTORSTORE
# =====================================================

@st.cache_resource
def load_vectorstore(embedder):

    if os.path.exists(VECTOR_STORE_PATH):

        with open(VECTOR_STORE_PATH, "rb") as f:

            return pickle.load(f)


    raw_docs = []


    if not os.path.exists(DOCS_DIR):

        st.error(f"Folder not found: {DOCS_DIR}")

        st.stop()


    # Load PDFs

    pdf_files = glob.glob(f"{DOCS_DIR}/*.pdf")

    txt_files = glob.glob(f"{DOCS_DIR}/*.txt")


    if len(pdf_files) == 0 and len(txt_files) == 0:

        st.error("No PDF or TXT files found in university_docs")

        st.stop()


    for file in pdf_files:

        st.info(f"Processing {os.path.basename(file)}")

        text = extract_pdf(file)

        if text:

            raw_docs.append(

                Document(
                    page_content=text,
                    metadata={"source": file}
                )
            )


    for file in txt_files:

        text = Path(file).read_text(
            encoding="utf-8",
            errors="ignore"
        )

        raw_docs.append(

            Document(
                page_content=clean_text(text),
                metadata={"source": file}
            )
        )


    splitter = RecursiveCharacterTextSplitter(

        chunk_size=500,
        chunk_overlap=100
    )


    docs = splitter.split_documents(raw_docs)


    vectorstore = FAISS.from_documents(
        docs,
        embedder
    )


    with open(VECTOR_STORE_PATH, "wb") as f:

        pickle.dump(vectorstore, f)


    return vectorstore


vectorstore = load_vectorstore(embedder)


# =====================================================
# CHAT SYSTEM
# =====================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


if "eval" not in st.session_state:

    st.session_state.eval = []


# Display history

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.markdown(msg["content"])


# Prompt template

prompt = ChatPromptTemplate.from_messages(

    [
        (
            "system",
            "Answer ONLY using the provided context."
        ),

        (
            "human",
            "{input}"
        ),
    ]
)


chain = prompt | llm | StrOutputParser()


# =====================================================
# USER INPUT
# =====================================================

if question := st.chat_input("Ask a question"):

    st.session_state.messages.append(

        {
            "role": "user",
            "content": question
        }
    )


    retriever = vectorstore.as_retriever(

        search_kwargs={"k": 3}
    )


    docs = retriever.invoke(question)


    context = "\n\n".join(

        d.page_content
        for d in docs
    )


    final_prompt = f"""

Context:
{context}

Question:
{question}

"""


    with st.chat_message("assistant"):

        placeholder = st.empty()

        answer = ""


        for chunk in chain.stream(

            {"input": final_prompt}

        ):

            answer += chunk

            placeholder.markdown(answer + "▌")


        placeholder.markdown(answer)


    st.session_state.messages.append(

        {
            "role": "assistant",
            "content": answer
        }
    )


# =====================================================
# EVALUATION SECTION
# =====================================================

st.divider()

st.subheader("Evaluation")


if len(st.session_state.messages) >= 2:

    col1, col2 = st.columns(2)


    with col1:

        correct = st.radio(

            "Correct?",

            ["Yes", "No"],

            key="correctness"
        )


    with col2:

        note = st.text_input(

            "Notes",
            key="note"
        )


    if st.button("Save Evaluation"):

        st.session_state.eval.append(

            {
                "question":
                st.session_state.messages[-2]["content"],

                "answer":
                st.session_state.messages[-1]["content"],

                "correct":
                correct,

                "note":
                note,
            }
        )

        st.success("Saved")


# =====================================================
# EXPORT CSV
# =====================================================

if st.session_state.eval:

    df = pd.DataFrame(

        st.session_state.eval
    )


    csv = df.to_csv(

        index=False
    ).encode("utf-8")


    st.download_button(

        "Download CSV",

        csv,

        "evaluation.csv",

        "text/csv"
    )