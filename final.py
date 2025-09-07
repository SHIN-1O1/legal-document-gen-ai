#!/usr/bin/env python3
"""
app.py — Integrated Legal AI Assistant (single-file)

Features:
 - Dual FAISS vectorstores: VertexAI embeddings + Legal-BERT (HuggingFace)
 - Chat via ChatVertexAI (Gemini) with compact prompts
 - Profile persistence (profile.json) extracted only from user messages
 - Memory buffer + summarization saved as searchable memory entries (embedding-backed)
 - Token monitoring (tiktoken + HF tokenizer) with GUI warnings
 - Optional web fallback (DuckDuckGo HTML scraping) when local context missing
 - Threaded indexing and search for responsive Tkinter GUI
"""

import os
import sys
import json
import hashlib
import logging
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import queue
import time
import re
import html
import urllib.parse

# GUI
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

# OCR / PDF
from PIL import Image, ImageEnhance
import pytesseract
from PyPDF2 import PdfReader

# language detection
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 0

# LangChain & embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_vertexai import VertexAIEmbeddings, ChatVertexAI
from langchain.docstore.document import Document

# tokenizers
try:
    import tiktoken
except Exception:
    tiktoken = None
from transformers import AutoTokenizer

# light web requests
import requests
from bs4 import BeautifulSoup  # optional (pip install beautifulsoup4)

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
@dataclass
class AppConfig:
    # your Vertex credentials json path (update if needed)
    GOOGLE_APPLICATION_CREDENTIALS: str = r"D:\\GEN_AI_25\\gemini-api-key.json"

    # storage
    DATA_DIR: str = "data"
    CACHE_DIR: str = "cache"
    MEMORY_DIR: str = "memory"
    PROFILE_PATH: str = "profile.json"

    # chunking / indexing
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    MIN_CHUNK_LENGTH: int = 50
    MAX_CHUNKS: int = 200

    TOP_K: int = 5
    DEDUP_THRESHOLD: float = 0.8

    # LLM / embeddings
    MODEL_NAME: str = "gemini-2.5-flash-lite"
    TEMPERATURE: float = 0.2
    MAX_OUTPUT_TOKENS: int = 1024

    # memory config
    MEMORY_TOP_K: int = 3
    MEMORY_TRUNCATE_CHARS: int = 400
    CONTEXT_TRUNCATE_CHARS: int = 900

    # faiss save interval (reduce I/O)
    FAISS_SAVE_INTERVAL: int = 6

    # indexing worker count
    INDEX_THREADS: int = 2

    # web fallback
    USE_WEB_FALLBACK: bool = True
    MAX_WEB_RESULTS: int = 3
    WEB_TIMEOUT: int = 6  # seconds

    # token limits for warnings
    LLM_TOKEN_WARN_RATIO: float = 0.8
    EMB_TOKEN_WARN_RATIO: float = 0.9
    EMB_MAX_TOKENS: int = 512  # for Legal-BERT

    def __post_init__(self):
        Path(self.DATA_DIR).mkdir(parents=True, exist_ok=True)
        Path(self.CACHE_DIR).mkdir(parents=True, exist_ok=True)
        Path(self.MEMORY_DIR).mkdir(parents=True, exist_ok=True)

CONFIG = AppConfig()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CONFIG.GOOGLE_APPLICATION_CREDENTIALS

# -----------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------
def setup_logging() -> logging.Logger:
    Path(CONFIG.DATA_DIR).mkdir(exist_ok=True)
    log_file = Path(CONFIG.DATA_DIR) / f"legal_ai_{datetime.now():%Y%m%d}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("legal-ai")

logger = setup_logging()

# -----------------------------------------------------------------------------
# UTILITIES
# -----------------------------------------------------------------------------
def file_fingerprint(path: str) -> str:
    st = Path(path).stat()
    s = f"{path}|{st.st_mtime_ns}|{st.st_size}"
    return hashlib.md5(s.encode()).hexdigest()

def compact_profile_for_prompt(profile: Dict[str, Any]) -> Dict[str, str]:
    # return only non-empty top-level keys (exclude internal extras) to save tokens
    return {k: str(v) for k, v in profile.items() if v and k != "_extra"}

# -----------------------------------------------------------------------------
# TOKEN MONITOR
# -----------------------------------------------------------------------------
class TokenMonitor:
    def __init__(self):
        # tiktoken for LLM token estimates
        self.enc = None
        if tiktoken:
            try:
                self.enc = tiktoken.get_encoding("cl100k_base")
            except Exception:
                self.enc = None
        # HF tokenizer for Legal-BERT token counting
        try:
            self.bert_tok = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
        except Exception:
            self.bert_tok = None

    def count_llm_tokens(self, text: str) -> int:
        if not text:
            return 0
        if self.enc:
            try:
                return len(self.enc.encode(text))
            except Exception:
                pass
        # fallback rough estimate: avg 1.3 tokens per word
        return max(1, int(len(text.split()) * 1.3))

    def count_bert_tokens(self, text: str) -> int:
        if not text:
            return 0
        if self.bert_tok:
            try:
                toks = self.bert_tok.encode(text, add_special_tokens=False)
                return len(toks)
            except Exception:
                pass
        return max(1, int(len(text.split()) * 1.0))

    def warnings_for(self, llm_tokens: int, emb_tokens: int) -> List[str]:
        warns = []
        if llm_tokens > CONFIG.LLM_TOKEN_WARN_RATIO * CONFIG.MAX_OUTPUT_TOKENS:
            warns.append(f"⚠️ LLM tokens high: {llm_tokens}/{CONFIG.MAX_OUTPUT_TOKENS}")
        if emb_tokens > CONFIG.EMB_TOKEN_WARN_RATIO * CONFIG.EMB_MAX_TOKENS:
            warns.append(f"⚠️ Embedding tokens high: {emb_tokens}/{CONFIG.EMB_MAX_TOKENS}")
        return warns

TOKEN_MON = TokenMonitor()

# -----------------------------------------------------------------------------
# EXTRACTOR: PDF & IMAGE OCR with caching
# -----------------------------------------------------------------------------
class Extractor:
    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)

    def from_pdf(self, pdf_path: str) -> str:
        fid = file_fingerprint(pdf_path)
        cpath = self.cache_dir / f"extract_{fid}.txt"
        if cpath.exists():
            return cpath.read_text(encoding="utf-8")

        text = ""
        try:
            with open(pdf_path, "rb") as f:
                reader = PdfReader(f)
                parts: List[str] = []
                for i, page in enumerate(reader.pages, 1):
                    try:
                        txt = (page.extract_text() or "").strip()
                    except Exception:
                        txt = ""
                    if txt:
                        parts.append(f"\n--- Page {i} ---\n{txt}")
                text = "\n".join(parts).strip()
        except Exception:
            logger.exception("Error reading PDF %s", pdf_path)
            text = ""

        if not text:
            text = "No readable text found."
        try:
            cpath.write_text(text, encoding="utf-8")
        except Exception:
            logger.exception("Failed to cache PDF extraction %s", cpath)
        return text

    def from_image(self, image_path: str, lang: str = "eng") -> str:
        fid = file_fingerprint(image_path)
        cpath = self.cache_dir / f"extract_{fid}.txt"
        if cpath.exists():
            return cpath.read_text(encoding="utf-8")

        text = ""
        try:
            with Image.open(image_path) as im:
                if im.mode != "L":
                    im = im.convert("L")
                im = ImageEnhance.Contrast(im).enhance(1.6)
                im = ImageEnhance.Sharpness(im).enhance(1.8)
                text = pytesseract.image_to_string(im, config="--oem 3 --psm 6", lang=lang) or ""
        except Exception:
            logger.exception("Image OCR error for %s", image_path)
            text = ""

        if not text:
            text = "No readable text found."
        try:
            cpath.write_text(text, encoding="utf-8")
        except Exception:
            logger.exception("Failed to cache image extraction %s", cpath)
        return text

# -----------------------------------------------------------------------------
# DUAL VECTOR STORE: Vertex + Legal-BERT using FAISS
# -----------------------------------------------------------------------------
class DualVectorStore:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Vertex embeddings (may fail if quota/credentials missing)
        self.vertex_emb = None
        try:
            self.vertex_emb = VertexAIEmbeddings(model_name="text-embedding-005")
            logger.info("VertexAIEmbeddings initialized.")
        except Exception:
            logger.exception("VertexAIEmbeddings init failed; vertex embeddings disabled.")
            self.vertex_emb = None

        # Initialize Legal-BERT embeddings (HuggingFace)
        self.legal_emb = None
        try:
            self.legal_emb = HuggingFaceEmbeddings(
                model_name="nlpaueb/legal-bert-base-uncased",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            logger.info("HuggingFace legal-bert embeddings initialized.")
        except Exception:
            logger.exception("HuggingFaceEmbeddings init failed; legal embeddings disabled.")
            self.legal_emb = None

        # FAISS stores
        self.vertex_store: Optional[FAISS] = None
        self.legal_store: Optional[FAISS] = None

        # lock and counters
        self._lock = threading.RLock()
        self._vertex_add_count = 0
        self._legal_add_count = 0

    def _vs_path(self, fid: str, kind: str) -> Path:
        return self.data_dir / f"vs_{kind}_{fid}"

    def build_or_load(self, text: Optional[str], fid: str) -> None:
        splitter = RecursiveCharacterTextSplitter(chunk_size=CONFIG.CHUNK_SIZE, chunk_overlap=CONFIG.CHUNK_OVERLAP)
        docs: List[Document] = []
        if text:
            docs = splitter.create_documents([text])
            docs = [d for d in docs if len(d.page_content.strip()) >= CONFIG.MIN_CHUNK_LENGTH]
            if len(docs) > CONFIG.MAX_CHUNKS:
                docs = docs[:CONFIG.MAX_CHUNKS]

        v_path = self._vs_path(fid, "vertex")
        with self._lock:
            try:
                if self.vertex_emb:
                    if v_path.exists():
                        self.vertex_store = FAISS.load_local(v_path.as_posix(), self.vertex_emb, allow_dangerous_deserialization=True)
                        logger.info("Loaded vertex store %s", v_path.name)
                    else:
                        self.vertex_store = FAISS.from_documents(docs or [Document(page_content="seed")], self.vertex_emb)
                        self.vertex_store.save_local(v_path.as_posix())
                        logger.info("Created vertex store %s", v_path.name)
            except Exception:
                logger.exception("Vertex store error for %s", fid)
                self.vertex_store = None

        l_path = self._vs_path(fid, "legal")
        with self._lock:
            try:
                if self.legal_emb:
                    if l_path.exists():
                        self.legal_store = FAISS.load_local(l_path.as_posix(), self.legal_emb, allow_dangerous_deserialization=True)
                        logger.info("Loaded legal store %s", l_path.name)
                    else:
                        self.legal_store = FAISS.from_documents(docs or [Document(page_content="seed")], self.legal_emb)
                        self.legal_store.save_local(l_path.as_posix())
                        logger.info("Created legal store %s", l_path.name)
            except Exception:
                logger.exception("Legal store error for %s", fid)
                self.legal_store = None

    def add_doc_to_stores(self, doc: Document, fid: str):
        with self._lock:
            if self.vertex_store:
                try:
                    self.vertex_store.add_texts([doc.page_content])
                    self._vertex_add_count += 1
                    if self._vertex_add_count % CONFIG.FAISS_SAVE_INTERVAL == 0:
                        self.vertex_store.save_local(self._vs_path(fid, "vertex").as_posix())
                except Exception:
                    logger.exception("Error adding to vertex store")
            if self.legal_store:
                try:
                    self.legal_store.add_texts([doc.page_content])
                    self._legal_add_count += 1
                    if self._legal_add_count % CONFIG.FAISS_SAVE_INTERVAL == 0:
                        self.legal_store.save_local(self._vs_path(fid, "legal").as_posix())
                except Exception:
                    logger.exception("Error adding to legal store")

    def search(self, query: str, k: int) -> List[str]:
        results = []
        with self._lock:
            if self.vertex_store:
                try:
                    vs = self.vertex_store.similarity_search_with_score(query, k=k*3)
                    results += [(doc.page_content, score) for doc, score in vs]
                except Exception:
                    logger.exception("Vertex similarity search error")
            if self.legal_store:
                try:
                    ls = self.legal_store.similarity_search_with_score(query, k=k*3)
                    results += [(doc.page_content, score) for doc, score in ls]
                except Exception:
                    logger.exception("Legal similarity search error")

        # Deduplication
        def simple_sig(text: str, nwords: int = 12) -> str:
            return " ".join(text.lower().split()[:nwords])

        seen_sigs = []
        unique_results = []
        for text, score in sorted(results, key=lambda x: x[1], reverse=True):
            sig = simple_sig(text)
            if any(sig == s for s in seen_sigs):
                continue
            # jaccard check against seen_sigs' texts (approx)
            def jaccard(a: str, b: str) -> float:
                A, B = set(a.lower().split()), set(b.lower().split())
                return len(A & B) / max(1, len(A | B))
            if any(jaccard(text, stext) >= CONFIG.DEDUP_THRESHOLD for stext in seen_sigs):
                continue
            seen_sigs.append(text)
            unique_results.append(text)
            if len(unique_results) >= k:
                break
        return unique_results

# -----------------------------------------------------------------------------
# PROFILE MEMORY
# -----------------------------------------------------------------------------
class ProfileMemory:
    SAFE_KEYS = {"name", "age", "gender", "location", "profession", "organization", "interests", "email", "phone"}

    def __init__(self, path: Path):
        self.path = Path(path)
        if not self.path.exists():
            self._write({})
        self._data = self._read()

    def _read(self) -> Dict[str, str]:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            logger.exception("Failed to read profile.json")
            return {}

    def _write(self, data: Dict[str, str]):
        try:
            self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            logger.exception("Failed to write profile.json")

    def set(self, key: str, value: str):
        key = key.strip()
        if key in self.SAFE_KEYS:
            self._data[key] = value
            self._write(self._data)
        else:
            self._data.setdefault("_extra", {})[key] = value
            self._write(self._data)

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        return self._data.get(key, default)

    def all(self) -> Dict[str, str]:
        return dict(self._data)

# -----------------------------------------------------------------------------
# MEMORY STORE
# -----------------------------------------------------------------------------
class MemoryStore:
    def __init__(self, mem_dir: Path, vs: Optional[DualVectorStore] = None):
        self.mem_dir = Path(mem_dir)
        self.mem_dir.mkdir(parents=True, exist_ok=True)
        self.vs = vs or DualVectorStore(self.mem_dir)
        self.fid = "chat_memory"
        self.buffer: List[Dict[str, Any]] = []
        self.buffer_limit = 12
        self.summarize_batch = 8
        try:
            self.vs.build_or_load("__seed__", self.fid)
        except Exception:
            logger.exception("Failed to initialize memory vector stores")

    def _truncated(self, role: str, content: str) -> str:
        return f"{role}: {content[:CONFIG.MEMORY_TRUNCATE_CHARS]}"

    def add(self, role: str, content: str):
        entry = {"role": role, "text": content, "ts": datetime.utcnow().isoformat()}
        self.buffer.append(entry)
        short = self._truncated(role, content)
        try:
            if self.vs.vertex_store:
                self.vs.vertex_store.add_texts([short])
            if self.vs.legal_store:
                self.vs.legal_store.add_texts([short])
        except Exception:
            logger.exception("Error adding truncated memory to vectorstores")

        if len(self.buffer) > self.buffer_limit:
            try:
                t = threading.Thread(target=self._summarize_and_persist_oldest, daemon=True)
                t.start()
            except Exception:
                logger.exception("Failed to start summarization thread")

    def _summarize_and_persist_oldest(self):
        if len(self.buffer) <= 0:
            return
        n = min(self.summarize_batch, len(self.buffer))
        to_summarize = self.buffer[:n]
        conv_lines = [f"{e['role']}: {e['text']}" for e in to_summarize]
        conv_blob = "\n".join(conv_lines)

        summarization_prompt = (
            "Summarize the conversation below into one short, factual sentence capturing the user's intent or key decision. "
            "Do not include personal identifiers.\n\n{text}"
        )

        summary = None
        try:
            temp_llm = ChatVertexAI(model=CONFIG.MODEL_NAME, temperature=0.0, max_output_tokens=140)
            pr = PromptTemplate(input_variables=["text"], template=summarization_prompt)
            chain = LLMChain(llm=temp_llm, prompt=pr)
            summary = chain.run({"text": conv_blob})
            logger.info("Memory summary generated.")
        except Exception:
            logger.exception("Vertex summarization failed; using fallback short capture")
            summary = " ; ".join(conv_lines[:2])[:200]

        summary = (summary or "No summary generated.").strip()
        label = f"MEMORY_SUMMARY: {summary}"

        try:
            if self.vs.vertex_store:
                self.vs.vertex_store.add_texts([label])
                self.vs.vertex_store.save_local(self.vs._vs_path(self.fid, "vertex").as_posix())
            if self.vs.legal_store:
                self.vs.legal_store.add_texts([label])
                self.vs.legal_store.save_local(self.vs._vs_path(self.fid, "legal").as_posic())
        except Exception:
            # note: fallback if .as_posic() typo? ensure correct method - use as_posix()
            try:
                if self.vs.vertex_store:
                    self.vs.vertex_store.save_local(self.vs._vs_path(self.fid, "vertex").as_posix())
                if self.vs.legal_store:
                    self.vs.legal_store.save_local(self.vs._vs_path(self.fid, "legal").as_posix())
            except Exception:
                logger.exception("Error writing memory summary to vectorstores")

        # remove summarized items but keep last one as stubble
        self.buffer = self.buffer[n - 1 :]

    def retrieve(self, query: str, k: int) -> List[str]:
        results = []
        try:
            results = self.vs.search(query, k=k)
        except Exception:
            logger.exception("MemoryStore retrieve error from vs.search")
            results = []

        recent_contexts = []
        for e in reversed(self.buffer[-CONFIG.MEMORY_TOP_K :]):
            recent_contexts.append(f"{e['role']}: {e['text']}")

        merged = recent_contexts + results
        seen = set()
        final = []
        for item in merged:
            key = item.strip()[:400]
            if key not in seen:
                final.append(item)
                seen.add(key)
            if len(final) >= k:
                break
        return final

# -----------------------------------------------------------------------------
# WEB FALLBACK (DuckDuckGo HTML scraping) - optional, enabled by config flag
# -----------------------------------------------------------------------------
def duckduckgo_search_snippets(query: str, max_results: int = 3, timeout: int = 6) -> List[str]:
    # We use DuckDuckGo HTML (no JS), parse titles/snippets
    try:
        base = "https://html.duckduckgo.com/html/"
        params = {"q": query}
        resp = requests.post(base, data=params, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code != 200:
            return []
        soup = BeautifulSoup(resp.text, "html.parser")
        snippets = []
        results = soup.find_all("a", {"class": "result__a"})
        containers = soup.find_all("div", {"class": "result__snippet"})
        # prefer snippet containers; fallback to link text
        for i in range(min(max_results, max(len(containers), len(results)))):
            s = ""
            if i < len(containers):
                s = containers[i].get_text(separator=" ", strip=True)
            elif i < len(results):
                s = results[i].get_text(separator=" ", strip=True)
            s = html.unescape(s)
            if s:
                snippets.append(s)
            if len(snippets) >= max_results:
                break
        return snippets
    except Exception:
        logger.exception("DuckDuckGo web fetch failed")
        return []

# -----------------------------------------------------------------------------
# QA ENGINE (multilingual, profile-safe, web-fallback optional, token monitoring)
# -----------------------------------------------------------------------------
class QAEngine:
    def __init__(self, profile: ProfileMemory, use_web: bool = CONFIG.USE_WEB_FALLBACK):
        self.profile = profile
        self.use_web = use_web
        self.llm = None
        try:
            self.llm = ChatVertexAI(model=CONFIG.MODEL_NAME, temperature=CONFIG.TEMPERATURE, max_output_tokens=CONFIG.MAX_OUTPUT_TOKENS)
            logger.info("ChatVertexAI initialized for QA.")
        except Exception:
            logger.exception("ChatVertexAI init failed; LLM disabled.")
            self.llm = None

        self.prompt = PromptTemplate(
            input_variables=["context", "history", "profile", "question", "lang_instruct", "web_results"],
            template=(
                "{lang_instruct}\n"
                "You are a concise, accurate assistant. Use PROFILE for personalization; do not assume document entities are the user's identity.\n\n"
                "PROFILE:\n{profile}\n\n"
                "PAST (short):\n{history}\n\n"
                "DOCUMENT SNIPPETS:\n{context}\n\n"
                "WEB RESULTS (if any):\n{web_results}\n\n"
                "QUESTION:\n{question}\n\n"
                "Answer clearly and succinctly."
            ),
        )

        self.extraction_prompt = PromptTemplate(
            input_variables=["text"],
            template=(
                "Extract only explicit personal info from this user message. Return JSON with keys among: "
                "name, age, gender, location, profession, organization, interests, email, phone. If none, return {}.\n\n"
                "Message:\n{text}\n"
            ),
        )

        self._chain_cache: Dict[str, LLMChain] = {}

    def _get_chain(self, prompt: PromptTemplate) -> LLMChain:
        key = prompt.template
        if key in self._chain_cache:
            return self._chain_cache[key]
        if not self.llm:
            raise RuntimeError("LLM not initialized")
        chain = LLMChain(llm=self.llm, prompt=prompt)
        self._chain_cache[key] = chain
        return chain

    def _lang_instruction(self, lang_code: str) -> str:
        mapping = {
            "en": "Respond in English.",
            "hi": "हिंदी में उत्तर दें।",
            "es": "Responde en Español.",
            "fr": "Répondez en Français.",
            "de": "Antworten Sie auf Deutsch.",
            "bn": "বাংলায় উত্তর দিন।",
            "ta": "தமிழில் பதில் அளியுங்கள்.",
            "ru": "Ответьте на русском языке.",
            "ar": "أجب باللغة العربية.",
        }
        return mapping.get(lang_code, "Respond in the user's language if possible; otherwise English.")

    def extract_profile_info(self, user_input: str):
        # Use LLM extraction when available; fallback to regex heuristics
        if not user_input or not user_input.strip():
            return
        # Do not extract from documents; only call this when user message is the input
        if self.llm:
            try:
                chain = self._get_chain(self.extraction_prompt)
                resp_text = chain.run({"text": user_input})
                try:
                    data = json.loads(resp_text)
                    if isinstance(data, dict):
                        for k, v in data.items():
                            if v is None:
                                continue
                            if not isinstance(v, (str, int, float)):
                                v = json.dumps(v, ensure_ascii=False)
                            self.profile.set(str(k), str(v))
                        logger.info("Profile updated from user input via LLM extractor: %s", list(data.keys()))
                        return
                except Exception:
                    logger.debug("LLM profile extraction non-JSON output: %s", resp_text[:200])
            except Exception:
                logger.exception("Profile extraction LLM error (falling back)")

        # Fallback heuristics (simple)
        txt = user_input.strip()
        lowered = txt.lower()
        # name heuristics: "i am X", "i'm X", "my name is X"
        name_match = re.search(r"\b(?:i am|i'm|my name is)\s+([A-Z][a-zA-Z]{1,40})", txt)
        if not name_match:
            # try capitalized token after I am ignoring start-of-sentence lowercases
            nm = re.search(r"(?:i am|i'm|my name is)\s+([^\s,\.]+)", lowered)
            if nm:
                candidate = nm.group(1).strip()
                candidate = candidate.capitalize()
                self.profile.set("name", candidate)
        else:
            self.profile.set("name", name_match.group(1).strip())

        # location heuristics: common city names or "i live in <place>"
        loc_match = re.search(r"\b(?:i live in|i'm in|i am in|i live at)\s+([A-Za-z\s]{2,60})", lowered)
        if loc_match:
            candidate = loc_match.group(1).strip().title()
            self.profile.set("location", candidate)
        else:
            # detect city-like words (Chennai, Mumbai, Delhi etc.)
            for city in ["chennai", "mumbai", "delhi", "bangalore", "kolkata", "hyderabad"]:
                if city in lowered:
                    self.profile.set("location", city.title())
                    break

    def detect_user_language(self, text: str) -> str:
        if not text:
            return "en"
        if len(text) < 10:
            return "en"
        try:
            return detect(text)
        except Exception:
            return "en"

    def _get_web_context(self, query: str) -> str:
        if not CONFIG.USE_WEB_FALLBACK or not self.use_web:
            return ""
        snippets = duckduckgo_search_snippets(query, max_results=CONFIG.MAX_WEB_RESULTS, timeout=CONFIG.WEB_TIMEOUT)
        return "\n".join(snippets)

    def answer(self, question: str, docs: List[str], memory: MemoryStore) -> Tuple[str, List[str]]:
        # language detection
        lang_code = self.detect_user_language(question or "")
        lang_instruct = self._lang_instruction(lang_code)

        # extract profile only from user messages
        try:
            self.extract_profile_info(question)
        except Exception:
            logger.exception("Profile extraction error in answer()")

        # compact profile
        compact_profile = compact_profile_for_prompt(self.profile.all())
        profile_json = json.dumps(compact_profile, ensure_ascii=False)

        # history + memory
        history_items = memory.retrieve(question, CONFIG.MEMORY_TOP_K)
        history = "\n".join(h[:CONFIG.CONTEXT_TRUNCATE_CHARS] for h in history_items)

        # doc context truncated
        truncated_docs = [d[:CONFIG.CONTEXT_TRUNCATE_CHARS] for d in docs] if docs else []
        context = "\n".join(truncated_docs) if truncated_docs else "No relevant document snippets."

        # web fallback context only if docs insufficient
        web_results_text = ""
        if CONFIG.USE_WEB_FALLBACK and self.use_web:
            # if docs empty or short, attempt web fetch
            if not docs or sum(len(d) for d in docs) < 120:
                web_results_text = self._get_web_context(question)

        # token monitoring
        filled_prompt_preview = self.prompt.format(context=context, history=history, profile=profile_json, question=question, lang_instruct=lang_instruct, web_results=web_results_text)
        llm_tokens = TOKEN_MON.count_llm_tokens(filled_prompt_preview)
        emb_tokens = TOKEN_MON.count_bert_tokens(question)
        warnings = TOKEN_MON.warnings_for(llm_tokens, emb_tokens)

        if not self.llm:
            logger.error("LLM not initialized.")
            return ("LLM not initialized. Check Vertex credentials.", warnings)

        chain = self._get_chain(self.prompt)
        try:
            out = chain.run({
                "context": context,
                "history": history,
                "profile": profile_json,
                "question": question,
                "lang_instruct": lang_instruct,
                "web_results": web_results_text,
            })
            return (out.strip(), warnings)
        except Exception:
            logger.exception("LLM chain run error")
            try:
                # fallback direct invocation
                filled = self.prompt.format(context=context, history=history, profile=profile_json,
                                            question=question, lang_instruct=lang_instruct, web_results=web_results_text)
                out = self.llm.invoke(filled)
                txt = getattr(out, "content", str(out)).strip()
                return (txt, warnings)
            except Exception:
                logger.exception("LLM fallback invocation failed")
                return ("Failed to produce an answer.", warnings)

# -----------------------------------------------------------------------------
# GUI (ttk) with threaded search/indexing and token warnings displayed in status
# -----------------------------------------------------------------------------
class AppGUI:
    def __init__(self, root, vs: DualVectorStore, qa: QAEngine, memory: MemoryStore):
        self.root = root
        self.vs = vs
        self.qa = qa
        self.memory = memory

        self.root.title("⚖️ Legal AI Assistant")
        self.root.geometry("980x650")
        self.root.minsize(840, 520)

        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("TButton", font=("Segoe UI", 10), padding=6)
        style.configure("TLabel", font=("Segoe UI", 10))
        style.configure("TEntry", font=("Segoe UI", 10))

        # paned layout
        self.pw = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.pw.pack(fill=tk.BOTH, expand=True)
        self.sidebar = ttk.Frame(self.pw, padding=12)
        self.pw.add(self.sidebar, weight=0)
        self.main_frame = ttk.Frame(self.pw, padding=12)
        self.pw.add(self.main_frame, weight=1)

        # sidebar widgets
        ttk.Label(self.sidebar, text="📂 Documents", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0,6))
        ttk.Button(self.sidebar, text="Upload PDF", command=self.upload_pdf).pack(fill="x", pady=4)
        ttk.Button(self.sidebar, text="Upload Image (PNG/JPG)", command=self.upload_image).pack(fill="x", pady=4)
        ttk.Separator(self.sidebar, orient="horizontal").pack(fill="x", pady=10)

        ttk.Label(self.sidebar, text="🗣 Query", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0,6))
        self.query_entry = ttk.Entry(self.sidebar, width=36)
        self.query_entry.pack(fill="x", pady=4)
        ttk.Button(self.sidebar, text="Search", command=self.run_search_threaded).pack(fill="x", pady=8)

        ttk.Label(self.sidebar, text="Force language (optional):", font=("Segoe UI", 9)).pack(anchor="w", pady=(8,2))
        self.lang_var = tk.StringVar(value="")
        lang_choices = ["", "en", "hi", "es", "fr", "de", "bn", "ta", "ru", "ar"]
        self.lang_combo = ttk.Combobox(self.sidebar, values=lang_choices, textvariable=self.lang_var, state="readonly")
        self.lang_combo.pack(fill="x")
        ttk.Separator(self.sidebar, orient="horizontal").pack(fill="x", pady=10)

        ttk.Label(self.sidebar, text="Profile (persisted):", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0,6))
        self.profile_view = scrolledtext.ScrolledText(self.sidebar, height=10, width=36, wrap=tk.WORD, font=("Segoe UI", 9))
        self.profile_view.pack(fill="both", pady=4)
        self.profile_view.insert(tk.END, json.dumps(self.qa.profile.all(), ensure_ascii=False, indent=2))
        self.profile_view.configure(state=tk.DISABLED)
        ttk.Button(self.sidebar, text="Refresh Profile View", command=self.refresh_profile_view).pack(fill="x", pady=(6,0))

        # main results
        top_row = ttk.Frame(self.main_frame)
        top_row.pack(fill="x")
        ttk.Label(top_row, text="🔍 Results", font=("Segoe UI", 12, "bold")).pack(side=tk.LEFT, anchor="w")
        self.clear_btn = ttk.Button(top_row, text="Clear", command=self.clear_results)
        self.clear_btn.pack(side=tk.RIGHT)

        self.result_box = scrolledtext.ScrolledText(self.main_frame, wrap=tk.WORD, font=("Consolas", 11))
        self.result_box.pack(fill=tk.BOTH, expand=True, pady=8)

        # status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor="w", padding=4)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # indexing queue & workers
        self._index_queue = queue.Queue()
        self._start_index_workers()

    def _start_index_workers(self):
        def worker():
            while True:
                job = self._index_queue.get()
                if job is None:
                    break
                text, fid = job
                try:
                    logger.info("Index worker processing %s", fid)
                    self.vs.build_or_load(text, fid)
                    logger.info("Index worker finished %s", fid)
                except Exception:
                    logger.exception("Index worker error")
                finally:
                    self._index_queue.task_done()
        for _ in range(max(1, CONFIG.INDEX_THREADS)):
            t = threading.Thread(target=worker, daemon=True)
            t.start()

    # UI helpers
    def set_status(self, text: str):
        self.status_var.set(text)
        logger.info("STATUS: %s", text)

    def refresh_profile_view(self):
        self.profile_view.configure(state=tk.NORMAL)
        self.profile_view.delete(1.0, tk.END)
        self.profile_view.insert(tk.END, json.dumps(self.qa.profile.all(), ensure_ascii=False, indent=2))
        self.profile_view.configure(state=tk.DISABLED)
        self.set_status("Profile refreshed.")

    def clear_results(self):
        self.result_box.delete(1.0, tk.END)
        self.set_status("Results cleared.")

    # file upload
    def upload_pdf(self):
        file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
        if not file_path:
            return
        try:
            extractor = Extractor(Path(CONFIG.CACHE_DIR))
            text = extractor.from_pdf(file_path)
            fid = Path(file_path).stem
            self._index_queue.put((text, fid))
            messagebox.showinfo("Success", f"PDF '{fid}' enqueued for indexing.")
            self.set_status(f"PDF '{fid}' enqueued.")
        except Exception:
            logger.exception("PDF upload error")
            messagebox.showerror("Error", "Failed to enqueue PDF for indexing.")
            self.set_status("Error while uploading PDF.")

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if not file_path:
            return
        try:
            extractor = Extractor(Path(CONFIG.CACHE_DIR))
            text = extractor.from_image(file_path)
            fid = Path(file_path).stem
            self._index_queue.put((text, fid))
            messagebox.showinfo("Success", f"Image '{fid}' enqueued for indexing.")
            self.set_status(f"Image '{fid}' enqueued.")
        except Exception:
            logger.exception("Image upload error")
            messagebox.showerror("Error", "Failed to enqueue image for indexing.")
            self.set_status("Error while uploading image.")

    # search (threaded)
    def run_search_threaded(self):
        q = self.query_entry.get().strip()
        if not q:
            messagebox.showwarning("Input Error", "Please enter a query.")
            return
        threading.Thread(target=self._run_search, args=(q,), daemon=True).start()

    def _run_search(self, query: str):
        self.set_status("Searching...")
        try:
            self.memory.add("User", query)
            docs = self.vs.search(query, k=CONFIG.TOP_K)
            forced_lang = (self.lang_var.get() or "").strip()
            if forced_lang:
                query_for_llm = f"(Reply language: {forced_lang})\n\n{query}"
            else:
                query_for_llm = query

            ans, warnings = self.qa.answer(query_for_llm, docs, self.memory)
            self.memory.add("Assistant", ans)

            # display
            self.result_box.delete(1.0, tk.END)
            self.result_box.insert(tk.END, "Answer:\n\n" + ans + "\n\n")
            if docs:
                self.result_box.insert(tk.END, "-" * 60 + "\nDocument snippets used:\n\n")
                for i, d in enumerate(docs, 1):
                    snippet = (d[:1000] + "...") if len(d) > 1000 else d
                    self.result_box.insert(tk.END, f"[Doc {i}]: {snippet}\n\n")

            # profile refresh
            self.refresh_profile_view()

            # status + warnings
            if warnings:
                self.set_status(" ; ".join(warnings))
            else:
                self.set_status("Answer generated successfully.")
        except Exception:
            logger.exception("Search error")
            messagebox.showerror("Error", "An error occurred during search.")
            self.set_status("Error during search.")

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    logger.info("Starting Legal AI Assistant (integrated).")
    memory = MemoryStore(Path(CONFIG.MEMORY_DIR))
    profile = ProfileMemory(Path(CONFIG.PROFILE_PATH))
    vs = DualVectorStore(Path(CONFIG.DATA_DIR))
    qa = QAEngine(profile, use_web=CONFIG.USE_WEB_FALLBACK)

    logger.info("Profile loaded: %s", profile.all())

    root = tk.Tk()
    app = AppGUI(root, vs, qa, memory)
    root.mainloop()

if __name__ == "__main__":
    main()
