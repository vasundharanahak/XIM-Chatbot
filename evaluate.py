"""
evaluate.py — Envie Chatbot Evaluation System
=============================================
Component 1: Side-by-side comparison of Envie vs ChatGPT on XIM domain questions
Component 2: Retrieved chunks inspector — see exactly what docs were pulled and which chunk drove the answer

Usage:
    python evaluate.py

Requirements:
    pip install openai pandas tabulate colorama
"""

import os
import pickle
import time
import textwrap
from datetime import datetime

import pandas as pd
from tabulate import tabulate
from colorama import init, Fore, Style, Back
from openai import OpenAI

from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Init colorama for colored terminal output
init(autoreset=True)


# ─────────────────────────────────────────────
# CONFIG — set your keys here or via env vars
# ─────────────────────────────────────────────

NVIDIA_API_KEY  = os.environ.get("NVIDIA_API_KEY", "")
OPENAI_API_KEY  = os.environ.get("OPENAI_API_KEY", "")
VECTOR_STORE_PATH = os.path.join(os.path.abspath("."), "vectorstore.pkl")

# Default XIM evaluation questions
DEFAULT_QUESTIONS = [
    "What is the admission process for the MBA program at XIM University?",
    "What are the fee structures for postgraduate programs at XIM University?",
    "Who is the Vice Chancellor of XIM University?",
    "What programs does XIM University offer in computer science?",
    "What scholarships are available at XIM University?",
    "What is the campus life like at XIM University?",
    "What are the PhD admission requirements at XIM University?",
    "Tell me about XIM University's ranking and recognition.",
]

CHUNK_RETRIEVAL_K = 5   # number of chunks to retrieve and inspect


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def divider(char="─", color=Fore.BLUE, width=70):
    print(color + char * width)

def header(text, color=Fore.CYAN):
    print()
    divider("═", color)
    print(color + Style.BRIGHT + f"  {text}")
    divider("═", color)
    print()

def section(text, color=Fore.YELLOW):
    print()
    print(color + Style.BRIGHT + f"▶ {text}")
    divider("─", color)

def wrap(text, width=68, indent="    "):
    return "\n".join(
        textwrap.fill(line, width=width, initial_indent=indent, subsequent_indent=indent)
        for line in text.split("\n")
    )


# ─────────────────────────────────────────────
# LOAD VECTORSTORE
# ─────────────────────────────────────────────

def load_vectorstore():
    if not os.path.exists(VECTOR_STORE_PATH):
        print(Fore.RED + f"✗ vectorstore.pkl not found at {VECTOR_STORE_PATH}")
        print(Fore.RED + "  Run api.py first to build the vectorstore.")
        exit(1)
    with open(VECTOR_STORE_PATH, "rb") as f:
        return pickle.load(f)


# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────

def load_envie():
    llm = ChatNVIDIA(
        model="meta/llama-3.3-70b-instruct",
        temperature=0
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are Envie, a smart and friendly AI assistant for XIM University.
Answer based only on the provided context. Be concise and accurate.
Never fabricate facts. If the context doesn't contain the answer, say so clearly."""),
        ("human", "{input}"),
    ])
    chain = prompt | llm | StrOutputParser()
    return chain


def load_chatgpt():
    if not OPENAI_API_KEY:
        print(Fore.RED + "✗ OPENAI_API_KEY not set.")
        exit(1)
    return OpenAI(api_key=OPENAI_API_KEY)


# ─────────────────────────────────────────────
# GET ANSWERS
# ─────────────────────────────────────────────

def get_envie_answer(question: str, vectorstore, chain) -> tuple[str, list]:
    """Returns (answer, retrieved_docs)"""
    retriever = vectorstore.as_retriever(search_kwargs={"k": CHUNK_RETRIEVAL_K})
    docs = retriever.invoke(question)
    context = "\n\n".join(
        f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
        for d in docs
    )
    final_prompt = f"Context from XIM University documents:\n{context}\n\nQuestion: {question} XIM University"
    answer = chain.invoke({"input": final_prompt})
    return answer, docs


def get_chatgpt_answer(question: str, client) -> str:
    """Returns ChatGPT answer with no context (pure LLM knowledge)"""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Answer questions about XIM University as accurately as you can based on your training knowledge."
                },
                {
                    "role": "user",
                    "content": question
                }
            ],
            temperature=0,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[ChatGPT Error: {e}]"


# ─────────────────────────────────────────────
# COMPONENT 1: SIDE-BY-SIDE COMPARISON
# ─────────────────────────────────────────────

def run_comparison(questions: list[str], vectorstore, chain, gpt_client):
    header("COMPONENT 1 — ENVIE vs ChatGPT", Fore.CYAN)
    print(Fore.WHITE + "  Comparing answers on XIM University domain questions.")
    print(Fore.WHITE + f"  Questions: {len(questions)}  |  Model: meta/llama-3.3-70b + RAG vs GPT-3.5-turbo\n")

    results = []

    for i, question in enumerate(questions, 1):
        section(f"Q{i}: {question}", Fore.YELLOW)

        print(Fore.WHITE + "  Fetching Envie answer...", end=" ", flush=True)
        t1 = time.time()
        envie_answer, _ = get_envie_answer(question, vectorstore, chain)
        envie_time = round(time.time() - t1, 2)
        print(Fore.GREEN + f"done ({envie_time}s)")

        print(Fore.WHITE + "  Fetching ChatGPT answer...", end=" ", flush=True)
        t2 = time.time()
        gpt_answer = get_chatgpt_answer(question, gpt_client)
        gpt_time = round(time.time() - t2, 2)
        print(Fore.GREEN + f"done ({gpt_time}s)")

        # Print side by side
        print()
        print(Fore.BLUE + Style.BRIGHT + "  ┌─ ENVIE (RAG)" + Fore.BLUE + f"  [{envie_time}s]")
        print(Fore.WHITE + wrap(envie_answer))
        print()
        print(Fore.MAGENTA + Style.BRIGHT + "  ┌─ ChatGPT (No context)" + Fore.MAGENTA + f"  [{gpt_time}s]")
        print(Fore.WHITE + wrap(gpt_answer))
        print()

        results.append({
            "Question": question,
            "Envie Answer": envie_answer,
            "Envie Time (s)": envie_time,
            "ChatGPT Answer": gpt_answer,
            "ChatGPT Time (s)": gpt_time,
        })

        time.sleep(0.5)

    return results


# ─────────────────────────────────────────────
# COMPONENT 2: CHUNK INSPECTOR
# ─────────────────────────────────────────────

def run_chunk_inspector(questions: list[str], vectorstore, chain):
    header("COMPONENT 2 — RETRIEVED CHUNKS INSPECTOR", Fore.GREEN)
    print(Fore.WHITE + "  For each question, see exactly which chunks were retrieved,")
    print(Fore.WHITE + f"  which documents they came from, and the full context used.\n")

    chunk_results = []

    for i, question in enumerate(questions, 1):
        section(f"Q{i}: {question}", Fore.GREEN)

        answer, docs = get_envie_answer(question, vectorstore, chain)

        print(Fore.CYAN + f"  Retrieved {len(docs)} chunks:\n")

        for j, doc in enumerate(docs, 1):
            source   = doc.metadata.get("source", "unknown")
            label    = doc.metadata.get("label", "")
            doc_type = doc.metadata.get("type", "file")
            scraped  = doc.metadata.get("scraped_at", "")
            content  = doc.page_content.strip()

            # Color code by source type
            source_color = Fore.CYAN if doc_type == "web" else Fore.YELLOW

            print(source_color + Style.BRIGHT + f"  Chunk {j}")
            print(Fore.WHITE   + f"  Source   : {source}")
            if label:
                print(Fore.WHITE + f"  Label    : {label}")
            if scraped:
                print(Fore.WHITE + f"  Scraped  : {scraped[:19]}")
            print(Fore.WHITE + f"  Type     : {'🌐 Web scraped' if doc_type == 'web' else '📄 Local document'}")
            print(Fore.WHITE + f"  Length   : {len(content)} chars")
            print(Fore.WHITE + "  Content  :")
            print(Fore.LIGHTWHITE_EX + wrap(content[:400] + ("..." if len(content) > 400 else ""), width=66))
            print()

            chunk_results.append({
                "Question": question,
                "Chunk #": j,
                "Source": source,
                "Label": label,
                "Type": doc_type,
                "Length (chars)": len(content),
                "Content Preview": content[:200],
            })

        # Show final answer
        print(Fore.GREEN + Style.BRIGHT + "  ✦ Final Envie Answer:")
        print(Fore.LIGHTGREEN_EX + wrap(answer))
        print()
        divider("·", Fore.WHITE, 70)

    return chunk_results


# ─────────────────────────────────────────────
# EXPORT TO CSV
# ─────────────────────────────────────────────

def export_results(comparison_results, chunk_results):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Comparison CSV
    if comparison_results:
        comp_path = f"eval_comparison_{timestamp}.csv"
        pd.DataFrame(comparison_results).to_csv(comp_path, index=False)
        print(Fore.GREEN + f"  ✓ Comparison results saved → {comp_path}")

    # Chunks CSV
    if chunk_results:
        chunk_path = f"eval_chunks_{timestamp}.csv"
        pd.DataFrame(chunk_results).to_csv(chunk_path, index=False)
        print(Fore.GREEN + f"  ✓ Chunk inspection saved  → {chunk_path}")


# ─────────────────────────────────────────────
# INTERACTIVE MENU
# ─────────────────────────────────────────────

def get_questions() -> list[str]:
    print(Fore.CYAN + "\n  Question Set:")
    print(Fore.WHITE + "  [1] Use default XIM evaluation questions")
    print(Fore.WHITE + "  [2] Enter your own questions")
    print(Fore.WHITE + "  [3] Both (default + your own)\n")

    choice = input(Fore.YELLOW + "  Choose [1/2/3]: ").strip()

    questions = []

    if choice in ("1", "3"):
        questions += DEFAULT_QUESTIONS

    if choice in ("2", "3"):
        print(Fore.WHITE + "\n  Enter questions one per line. Empty line to finish:")
        while True:
            q = input(Fore.LIGHTWHITE_EX + "  > ").strip()
            if not q:
                break
            questions.append(q)

    if not questions:
        print(Fore.YELLOW + "  No questions entered, using defaults.")
        questions = DEFAULT_QUESTIONS

    return questions


def main():
    # ── Validate keys ──
    if not NVIDIA_API_KEY:
        print(Fore.RED + "\n✗ NVIDIA_API_KEY not set.")
        print(Fore.RED + "  Run: set NVIDIA_API_KEY=your_key")
        exit(1)
    if not OPENAI_API_KEY:
        print(Fore.RED + "\n✗ OPENAI_API_KEY not set.")
        print(Fore.RED + "  Run: set OPENAI_API_KEY=your_key")
        exit(1)

    # ── Splash ──
    header("ENVIE EVALUATION SYSTEM", Fore.CYAN)
    print(Fore.WHITE + "  XIM University Chatbot — Quality Assessment Tool")
    print(Fore.WHITE + f"  Vectorstore: {VECTOR_STORE_PATH}")
    print(Fore.WHITE + f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

    # ── Menu ──
    print(Fore.CYAN + "  What do you want to run?")
    print(Fore.WHITE + "  [1] Component 1 only  — Envie vs ChatGPT comparison")
    print(Fore.WHITE + "  [2] Component 2 only  — Retrieved chunks inspector")
    print(Fore.WHITE + "  [3] Both components")
    print(Fore.WHITE + "  [4] Exit\n")

    mode = input(Fore.YELLOW + "  Choose [1/2/3/4]: ").strip()
    if mode == "4":
        exit(0)

    # ── Load resources ──
    print(Fore.WHITE + "\n  Loading vectorstore...", end=" ", flush=True)
    vectorstore = load_vectorstore()
    print(Fore.GREEN + "done")

    print(Fore.WHITE + "  Loading Envie (NVIDIA)...", end=" ", flush=True)
    chain = load_envie()
    print(Fore.GREEN + "done")

    gpt_client = None
    if mode in ("1", "3"):
        print(Fore.WHITE + "  Loading ChatGPT (OpenAI)...", end=" ", flush=True)
        gpt_client = load_chatgpt()
        print(Fore.GREEN + "done")

    # ── Get questions ──
    questions = get_questions()
    print(Fore.WHITE + f"\n  Running evaluation on {len(questions)} question(s)...\n")

    comparison_results = []
    chunk_results = []

    # ── Run ──
    if mode in ("1", "3"):
        comparison_results = run_comparison(questions, vectorstore, chain, gpt_client)

    if mode in ("2", "3"):
        chunk_results = run_chunk_inspector(questions, vectorstore, chain)

    # ── Summary ──
    header("SUMMARY", Fore.CYAN)

    if comparison_results:
        avg_envie = round(sum(r["Envie Time (s)"] for r in comparison_results) / len(comparison_results), 2)
        avg_gpt   = round(sum(r["ChatGPT Time (s)"] for r in comparison_results) / len(comparison_results), 2)
        table = [
            ["Questions evaluated", len(comparison_results)],
            ["Avg Envie response time", f"{avg_envie}s"],
            ["Avg ChatGPT response time", f"{avg_gpt}s"],
        ]
        print(tabulate(table, tablefmt="rounded_outline"))
        print()

    if chunk_results:
        web_chunks  = sum(1 for r in chunk_results if r["Type"] == "web")
        file_chunks = sum(1 for r in chunk_results if r["Type"] != "web")
        unique_sources = len(set(r["Source"] for r in chunk_results))
        table = [
            ["Total chunks retrieved", len(chunk_results)],
            ["From web scraping", web_chunks],
            ["From local documents", file_chunks],
            ["Unique sources used", unique_sources],
        ]
        print(tabulate(table, tablefmt="rounded_outline"))
        print()

    # ── Export ──
    save = input(Fore.YELLOW + "  Export results to CSV? [y/n]: ").strip().lower()
    if save == "y":
        export_results(comparison_results, chunk_results)

    print(Fore.GREEN + Style.BRIGHT + "\n  ✓ Evaluation complete.\n")


if __name__ == "__main__":
    main()
