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

from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from transformers.utils import logging
logging.set_verbosity_error()

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

def evaluate_self_retrieval(vectorstore, k=8, sample_size=50):
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": 20}
    )

    # Get all stored documents (chunks)
    all_docs = vectorstore.docstore._dict.values()
    all_docs = list(all_docs)

    if sample_size:
        all_docs = all_docs[:sample_size]

    total_precision = 0
    total_recall = 0
    total_hits = 0

    for doc in all_docs:
        source = doc.metadata.get("source")
        
        # Use first 200 characters as query
        query = doc.page_content[:200]

        retrieved_docs = retriever.invoke(query)
        retrieved_sources = [d.metadata.get("source") for d in retrieved_docs]

        relevant_retrieved = retrieved_sources.count(source)

        precision = relevant_retrieved / k
        recall = 1 if relevant_retrieved > 0 else 0  # since 1 true source exists

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
    },
    {
        "question": "What facilities are available on campus for student learning and development?",
        "reference": "The university provides libraries, research centers, computer labs, classrooms with modern technology, sports facilities, and student clubs that help in academic and personal development."
    },
    {
        "question": "How safe is the campus for students, especially those staying in hostels?",
        "reference": "XIM University maintains campus security through surveillance systems, security personnel, and controlled access to hostels and academic buildings to ensure a safe environment for students."
    },
    {
        "question": "Are there internship opportunities available during the course programs?",
        "reference": "Many programs at XIM University include internship opportunities where students gain practical industry experience through summer internships, live projects, and collaborations with companies."
    },
    {
        "question": "What extracurricular activities and student organizations are available at XIM University?",
        "reference": "Students at XIM University can participate in cultural clubs, technical societies, entrepreneurship cells, sports teams, and student-run events that encourage leadership and teamwork."
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

#ROUGE
scorer = rouge_scorer.RougeScorer(
    ['rouge1', 'rouge2', 'rougeL'],
    use_stemmer=True
)


def evaluate_rouge(model_answers, evaluation_set):

    scores = []

    for i, item in enumerate(evaluation_set):

        reference = item["reference"]
        prediction = model_answers[i]

        score = scorer.score(reference, prediction)

        scores.append({
            "question": item["question"],
            "rouge1": score["rouge1"].fmeasure,
            "rouge2": score["rouge2"].fmeasure,
            "rougeL": score["rougeL"].fmeasure
        })

    return scores

model_answers = [
    "We offer a range of undergraduate and postgraduate programs. Our undergraduate programs include: B.Com in Accounting/Finance (Hons.) B.B.M in HRM/Marketing (Hons.) B.Sc. in Media Studies (Hons.) Our postgraduate programs include: Business Management Human Resource Management Rural Management Sustainability Management Urban Management and Governance M.Tech in CSE M.Sc in Economics M.A in Mass Communication M.Sc in Psychology M.A in English LLM We also offer doctoral programs and other specialized courses.",

"We have two campuses: our main campus is located in Nijigada Kurki, Harirajpur, Puri District, and we also have an additional campus at Xavier Square, Bhubaneswar. As for the campus environment, we take pride in providing a safe and conducive space for our students to learn and grow. Our campus features solar-powered and air-conditioned classrooms, labs, and multimedia halls with excellent ventilation and natural lighting. We also have a public address system and ergonomically designed facilities. Additionally, we have an active Student Life department that engages students outside of the classroom with various clubs, societies, music, arts, dance, and sports opportunities. Overall, we strive to create a vibrant and supportive community that fosters academic excellence and personal growth.",

"At XIM University, the admission process for MBA programs typically involves the following steps: Applicants need to appear for one of the following entrance exams: XAT, CAT, GMAT, or XGMT (conducted by XIM University). Registration for XAT, CAT, or GMAT is a prerequisite for applying to XIM University's MBA program, if not opting for XGMT. Shortlisting for MBA programs is based on test scores, academic credentials, and personal interview. For some programs like MBA RM, SM, UMG, PPM, admission interviews may be conducted without test scores, but for others, test scores are required. The selection process also considers academic performance, with top rank holders in any discipline being shortlisted under the rank holder category for some programs. It's best to check with the university directly for the most up-to-date information on the admission process and deadlines.",

"We have limited hostel facilities available, with preference given to students coming from outside Bhubaneswar. Here are some key points about our hostel facilities: Separate residential buildings for boys and girls. Each hostel has its own mess, serving both vegetarian and non-vegetarian meals, managed by a students' committee. Facilities include: 24-hour internet connectivity through WiFi, laser printing and photocopying machines, multipurpose hall, cooperative store, book club, washing machines, water coolers, geysers, telephones, televisions. Hostels are not compulsory for undergraduate programs, but local students can avail of them if available, after outside Odisha and outside Bhubaneswar students are accommodated.",

"We offer several scholarships and financial aid options for our students. Here are the details: Meritorious Scholarship: The topper of each university program at the end of the year will get a scholarship worth 50% course tuition fee waiver. Scholarship for SC/ST/Economically Backward/Minority students: 3 students from each year program will receive a 50% course tuition fee waiver. Scholarship for students with 85% marks in aggregate in class 10 and 12 exams: A 50% course tuition fee waiver will be provided at the time of admission, and this offer will continue if the student secures a minimum CGPA of 8.0 in each subsequent year. National Fellowship and Scholarship for Higher Education of ST students is also available, offered by the Ministry of Tribal Affairs.",

"We have a strong track record of placements. For instance, our MBA-UMG program for the 2023-25 batch achieved 100% placements. Similarly, our MMSBF program (formerly MBF) also achieved 100% placements in record time for the 2022-24 batch. Our students are well-equipped to assume responsible positions soon after graduation, thanks to our rigorous grounding in business fundamentals.",

"We have a range of facilities to support student learning and development, including: Air-conditioned classrooms with audio-visual facilities and LCD projectors, labs and multimedia halls with excellent ventilation and ample natural lighting, public address system, ergonomically designed classrooms, solar-powered facilities, undisturbed power supply, indoor games space, library and other academic resources. For student well-being, we have: a campus doctor and nurses, 24/7 ambulance for medical emergencies, CCTV surveillance and security personnel for safety, bank branch and ATM machines on campus. We also have facilities to support student comfort and convenience, such as washing machines, water coolers, geysers, telephones, televisions. Additionally, we encourage learning opportunities beyond the classroom through various student forums and a learning-by-doing model that promotes extensive interaction with all components of the university's ecosystem.",

"At XIM University, we take the safety of all our students seriously. The campus has a safe environment, with separate residential buildings (hostels) for boys and girls, each with its own mess and managed by a students' committee. We also have a 24-hour internet connectivity, laser printing, and photocopying machines, as well as other facilities like multipurpose halls, cooperative stores, and book clubs. To ensure student safety, we have rules in place, such as requiring prior permission from the warden/prefect and approval from the Hostel In-charges for students to leave campus. This helps us maintain a secure environment for all students, especially those staying in hostels.",

"Yes, we offer internship opportunities. In fact, our programs include four internships with planning organizations, which provide our students with hands-on experience and industry exposure. Additionally, we have a 68-week Planning Immersion module with government departments, which is a unique feature of our programs. These opportunities help our students become industry-ready and gain practical knowledge in their field of study.",

"We have a range of extracurricular activities and student organizations available on campus. These include: Music, Arts, Dance, Sports, Student-run clubs and societies. These activities provide our students with opportunities to engage outside of the classroom and develop new skills and interests."
]



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


evaluate_bertscore(vectorstore, evaluation_set)

results = evaluate_rouge(model_answers, evaluation_set)
for r in results:
    print(r)