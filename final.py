import os
import sys
import json
import time
import hashlib
import logging
import threading
import queue
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

# ---- Third-party deps ----
from PIL import Image, ImageEnhance
import pytesseract
import fitz  # PyMuPDF for hybrid PDF extraction
import io

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_vertexai import VertexAIEmbeddings, ChatVertexAI

# ---- GUI ----
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext

# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class AppConfig:
    GOOGLE_APPLICATION_CREDENTIALS: str = r"gemini-api-key.json"
    DATA_DIR: str = "data"
    CACHE_DIR: str = "cache"
    STORES_DIR: str = "stores"
    
    # RAG settings
    CHUNK_SIZE: int = 400
    CHUNK_OVERLAP: int = 40
    MIN_CHUNK_LENGTH: int = 50
    TOP_K: int = 4
    DEDUP_THRESHOLD: float = 0.78
    MODEL_NAME: str = "gemini-2.5-flash-lite"
    TEMPERATURE: float = 0.1
    MAX_OUTPUT_TOKENS: int = 2048
    INDEX_THREADS: int = 2
    
    # Memory settings
    SLIDING_WINDOW: int = 8

    def __post_init__(self):
        Path(self.DATA_DIR).mkdir(parents=True, exist_ok=True)
        Path(self.CACHE_DIR).mkdir(parents=True, exist_ok=True)
        Path(self.STORES_DIR).mkdir(parents=True, exist_ok=True)


CONFIG = AppConfig()
if CONFIG.GOOGLE_APPLICATION_CREDENTIALS and Path(CONFIG.GOOGLE_APPLICATION_CREDENTIALS).exists():
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = CONFIG.GOOGLE_APPLICATION_CREDENTIALS

# =============================================================================
# LOGGING
# =============================================================================

def setup_logging() -> logging.Logger:
    log_file = Path(CONFIG.DATA_DIR) / f"hybrid_{datetime.now():%Y%m%d}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("legal-hybrid")

logger = setup_logging()

# =============================================================================
# UTILS
# =============================================================================

def file_fingerprint(path: str) -> str:
    try:
        st = Path(path).stat()
        s = f"{path}|{st.st_mtime_ns}|{st.st_size}"
        return hashlib.md5(s.encode()).hexdigest()
    except Exception:
        return hashlib.md5(path.encode()).hexdigest()

# =============================================================================
# EXTRACTOR (with Hybrid PDF-Image OCR)
# =============================================================================

class Extractor:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir

    def _perform_ocr(self, image_bytes: bytes) -> str:
        """Helper function to perform OCR on image bytes."""
        text = ""
        try:
            with Image.open(io.BytesIO(image_bytes)) as im:
                if im.mode != "L": im = im.convert("L")
                im = ImageEnhance.Contrast(im).enhance(1.8)
                im = ImageEnhance.Sharpness(im).enhance(1.8)
                text = pytesseract.image_to_string(im, config="--oem 3 --psm 6", lang="eng") or ""
        except Exception as e:
            logger.error(f"OCR processing failed: {e}")
        return text.strip()

    def from_pdf(self, pdf_path: str) -> str:
        """Extracts both digital text and OCR'd text from images within a PDF."""
        fid = file_fingerprint(pdf_path)
        cpath = self.cache_dir / f"extract_{fid}.txt"
        if cpath.exists():
            return cpath.read_text(encoding="utf-8")
        
        text_parts = []
        try:
            with fitz.open(pdf_path) as doc:
                for i, page in enumerate(doc, start=1):
                    page_content = [f"\n--- Page {i} ---"]
                    
                    # 1. Extract digital text
                    digital_text = page.get_text().strip()
                    if digital_text:
                        page_content.append(digital_text)
                    
                    # 2. Extract images and perform OCR
                    image_list = page.get_images(full=True)
                    if image_list:
                        ocr_texts = []
                        for img_index, img in enumerate(image_list):
                            xref = img[0]
                            base_image = doc.extract_image(xref)
                            image_bytes = base_image["image"]
                            
                            ocr_text = self._perform_ocr(image_bytes)
                            if ocr_text:
                                ocr_texts.append(ocr_text)
                        
                        if ocr_texts:
                            page_content.append("\n--- OCR Text from Images ---\n" + "\n".join(ocr_texts))
                    
                    text_parts.append("\n".join(page_content))

        except Exception:
            logger.exception("Error reading PDF with PyMuPDF %s", pdf_path)
        
        text = "\n".join(text_parts).strip()
        if not text:
            text = "No readable text found."
            
        try:
            cpath.write_text(text, encoding="utf-8")
        except Exception:
            logger.exception("Failed to cache hybrid PDF extraction %s", cpath)
            
        return text

    def from_image(self, image_path: str) -> str:
        """Extracts text from a standalone image file using OCR."""
        fid = file_fingerprint(image_path)
        cpath = self.cache_dir / f"extract_{fid}.txt"
        if cpath.exists():
            return cpath.read_text(encoding="utf-8")
        
        try:
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            text = self._perform_ocr(image_bytes)
        except Exception:
            logger.exception("Image file reading error for %s", image_path)
            text = ""
        
        if not text.strip():
            text = "No readable text found."
            
        try:
            cpath.write_text(text, encoding="utf-8")
        except Exception:
            logger.exception("Failed to cache image extraction %s", cpath)
            
        return text

# =============================================================================
# DUAL VECTOR STORE
# =============================================================================

class DualVectorStore:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.vertex_emb, self.legal_emb = None, None
        self._lock = threading.RLock()
        
        self.doc_store_name = "unified_documents"
        self.doc_vertex_store: Optional[FAISS] = None
        self.doc_legal_store: Optional[FAISS] = None

        self._init_embeddings()
        self._load_or_create_unified_stores()

    def _init_embeddings(self):
        try:
            self.vertex_emb = VertexAIEmbeddings(model_name="text-embedding-004")
            logger.info("VertexAIEmbeddings initialized.")
        except Exception: logger.exception("VertexAIEmbeddings init failed.")
        try:
            self.legal_emb = HuggingFaceEmbeddings(model_name="nlpaueb/legal-bert-base-uncased", model_kwargs={"device": "cpu"}, encode_kwargs={"normalize_embeddings": True})
            logger.info("HuggingFace legal-bert embeddings initialized.")
        except Exception: logger.exception("HuggingFaceEmbeddings init failed.")

    def _load_or_create_unified_stores(self):
        with self._lock:
            # Vertex Store
            v_path = self.base_dir / f"vs_vertex_{self.doc_store_name}"
            if v_path.exists() and self.vertex_emb:
                logger.info("Loading existing unified Vertex document store...")
                self.doc_vertex_store = FAISS.load_local(v_path.as_posix(), self.vertex_emb, allow_dangerous_deserialization=True)
            elif self.vertex_emb:
                logger.info("Creating new unified Vertex document store.")
                self.doc_vertex_store = FAISS.from_texts(["init"], self.vertex_emb)

            # Legal Store
            l_path = self.base_dir / f"vs_legal_{self.doc_store_name}"
            if l_path.exists() and self.legal_emb:
                logger.info("Loading existing unified Legal-BERT document store...")
                self.doc_legal_store = FAISS.load_local(l_path.as_posix(), self.legal_emb, allow_dangerous_deserialization=True)
            elif self.legal_emb:
                logger.info("Creating new unified Legal-BERT document store.")
                self.doc_legal_store = FAISS.from_texts(["init"], self.legal_emb)

    def add_documents(self, docs: List[Document]):
        if not docs:
            return
        with self._lock:
            if self.doc_vertex_store:
                self.doc_vertex_store.add_documents(docs)
            if self.doc_legal_store:
                self.doc_legal_store.add_documents(docs)
        self.save_stores()

    def save_stores(self):
        with self._lock:
            if self.doc_vertex_store:
                v_path = self.base_dir / f"vs_vertex_{self.doc_store_name}"
                self.doc_vertex_store.save_local(v_path.as_posix())
            if self.doc_legal_store:
                l_path = self.base_dir / f"vs_legal_{self.doc_store_name}"
                self.doc_legal_store.save_local(l_path.as_posix())

        def search(self, query: str, k: int, doc_filter: Optional[str] = None) -> List[str]:
            results: List[Tuple[Document, float]] = []
            with self._lock:
                if self.doc_vertex_store:
                    results.extend(self.doc_vertex_store.similarity_search_with_score(query, k=max(1, k * 2)))
                if self.doc_legal_store:
                    results.extend(self.doc_legal_store.similarity_search_with_score(query, k=max(1, k * 2)))

            def jaccard(a: str, b: str) -> float:
                A, B = set(a.lower().split()), set(b.lower().split())
                return len(A & B) / max(1, len(A | B))

            seen_texts, unique_results = [], []
            for doc, score in sorted(results, key=lambda x: x[1]):
                source = doc.metadata.get('source', 'Unknown')

                if doc_filter and doc_filter.lower() not in source.lower():
                    continue

                page_content = doc.page_content
                if any(jaccard(page_content, s) >= CONFIG.DEDUP_THRESHOLD for s in seen_texts):
                    continue

                formatted_content = f"Source: {source}\n\n{page_content}"
                seen_texts.append(page_content)
                unique_results.append(formatted_content)
                if len(unique_results) >= k:
                    break
            return unique_results


# =============================================================================
# MEMORY STORE
# =============================================================================
class MemoryStore:
    def __init__(self):
        self.buffer: List[Dict[str, Any]] = []
        self.buffer_limit = CONFIG.SLIDING_WINDOW
        
    def add(self, role: str, content: str):
        entry = {"role": role, "text": content, "ts": datetime.now(timezone.utc).isoformat()}
        self.buffer.append(entry)
        
        while len(self.buffer) > self.buffer_limit:
            self.buffer.pop(0)

    def get_recent_history(self) -> List[str]:
        return [f"{e['role']}: {e['text']}" for e in self.buffer]

# =============================================================================
# QA ENGINE
# =============================================================================
class QAEngine:
    def __init__(self):
        self.llm = ChatVertexAI(
            model=CONFIG.MODEL_NAME,
            temperature=CONFIG.TEMPERATURE,
            max_output_tokens=CONFIG.MAX_OUTPUT_TOKENS,
        )
        self.prompt = PromptTemplate(
            input_variables=["history", "context", "question"],
            template=(
                "You are a helpful legal assistant. Use the conversation HISTORY and the provided CONTEXT from legal documents to answer the QUESTION.\n"
                "The CONTEXT sections are formatted with their source document name.\n\n"
                "HISTORY:\n{history}\n\n"
                "CONTEXT:\n{context}\n\n"
                "QUESTION: {question}\n\n"
                "Answer:"
            ),
        )

    def answer(self, docs: List[str], question: str, history_context: List[str]) -> str:
        ctx = "\n---\n".join(docs)
        hist = "\n".join(history_context) or "No history yet."
        filled_prompt = self.prompt.format(context=ctx, history=hist, question=question)
        
        try:
            resp = self.llm.invoke(filled_prompt)
            return resp.content.strip() if hasattr(resp, "content") else str(resp).strip()
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "I apologize, but I encountered an error. Please try again."

# =============================================================================
# INDEXING MANAGER
# =============================================================================
class IndexingManager:
    def __init__(self, vs: DualVectorStore):
        self.vs = vs
        self._index_queue = queue.Queue()
        self._workers = [threading.Thread(target=self._worker, daemon=True) for _ in range(CONFIG.INDEX_THREADS)]
        for w in self._workers: w.start()

    def enqueue(self, text: str, doc_name: str):
        self._index_queue.put((text, doc_name))

    def _worker(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=CONFIG.CHUNK_SIZE, chunk_overlap=CONFIG.CHUNK_OVERLAP)
        while True:
            text, doc_name = self._index_queue.get()
            if doc_name is None: break
            
            logger.info(f"Index worker processing '{doc_name}'")
            docs = splitter.create_documents([text], metadatas=[{"source": doc_name}])
            docs = [d for d in docs if len(d.page_content.strip()) >= CONFIG.MIN_CHUNK_LENGTH]

            if docs:
                self.vs.add_documents(docs)
            
            logger.info(f"Index worker finished '{doc_name}'")
            self._index_queue.task_done()

    def shutdown(self):
        logger.info("Shutting down index manager...")
        for _ in self._workers: self._index_queue.put((None, None))
        for w in self._workers: w.join(timeout=2)

# =============================================================================
# GUI APPLICATION (with Unified Uploader)
# =============================================================================
class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Legal Assistant — Unified Uploader")
        self.root.geometry("1000x700")

        self.extractor = Extractor(Path(CONFIG.CACHE_DIR))
        self.vs = DualVectorStore(Path(CONFIG.STORES_DIR))
        self.qa = QAEngine()
        self.memory = MemoryStore()
        self.indexer = IndexingManager(self.vs)
        
        self.indexed_files: Set[str] = set()
        self.busy = False
        
        self._setup_gui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def _setup_gui(self):
        top_frame = tk.Frame(self.root)
        top_frame.pack(fill="x", padx=8, pady=8)

        tk.Button(top_frame, text="Load Document", command=self.load_document).pack(side="left")
        tk.Button(top_frame, text="Clear Chat", command=self.clear_chat).pack(side="left", padx=6)

        self.status = tk.StringVar(value="Ready. Load a document to begin.")
        tk.Label(top_frame, textvariable=self.status, anchor="w").pack(side="left", padx=20)

        self.chat = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, height=25)
        self.chat.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.chat_insert_system("Loaded. Use 'Load Document' to add PDFs or images to the knowledge base.")

        bottom_frame = tk.Frame(self.root)
        bottom_frame.pack(fill="x", padx=8, pady=8)
        
        self.entry = tk.Text(bottom_frame, height=3)
        self.entry.pack(side="left", fill="both", expand=True)
        tk.Button(bottom_frame, text="Send", command=self.on_send).pack(side="left", padx=6)
        
        self.entry.bind("<Return>", self._enter_key_handler)

    def on_closing(self):
        if self.indexer: self.indexer.shutdown()
        self.root.destroy()

    def chat_insert(self, who: str, text: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.chat.configure(state="normal")
        self.chat.insert("end", f"[{ts}] {who}: {text}\n\n")
        self.chat.configure(state="disabled")
        self.chat.see("end")

    def chat_insert_system(self, text: str):
        self.chat_insert("SYSTEM", text)

    def _enter_key_handler(self, event):
        self.on_send()
        return "break"

    def set_busy(self, flag: bool, msg: str = ""):
        self.busy = flag
        self.status.set(msg or ("Working..." if flag else "Ready"))
        self.root.update_idletasks()

    def clear_chat(self):
        self.chat.configure(state="normal")
        self.chat.delete("1.0", "end")
        self.chat.configure(state="disabled")
        self.chat_insert_system("Chat cleared. Loaded documents remain in the knowledge base.")
        if self.memory: self.memory.buffer.clear()

    def load_document(self):
        if self.busy: return
        
        filetypes = [
            ("All Supported Files", "*.pdf *.png *.jpg *.jpeg"),
            ("PDF Documents", "*.pdf"),
            ("Image Files", "*.png *.jpg *.jpeg"),
            ("All Files", "*.*")
        ]
        path_str = filedialog.askopenfilename(title="Select a Document or Image", filetypes=filetypes)
        if not path_str: return

        path = Path(path_str)
        file_extension = path.suffix.lower()
        file_type = ""

        if file_extension == ".pdf":
            file_type = "pdf"
        elif file_extension in [".png", ".jpg", ".jpeg"]:
            file_type = "image"
        else:
            messagebox.showwarning("Unsupported File", f"The file type '{file_extension}' is not supported.")
            return
            
        threading.Thread(target=self._load_worker, args=(str(path), file_type), daemon=True).start()

    def _load_worker(self, path: str, file_type: str):
        fid = file_fingerprint(path)
        if fid in self.indexed_files:
            self.chat_insert_system(f"File '{Path(path).name}' has already been indexed in this session.")
            return

        self.set_busy(True, f"Extracting text from {Path(path).name}...")
        text = ""
        if file_type == "pdf":
            text = self.extractor.from_pdf(path)
        elif file_type == "image":
            text = self.extractor.from_image(path)
        
        try:
            if not text or text.startswith("No readable text") or len(text.strip()) < CONFIG.MIN_CHUNK_LENGTH:
                self.chat_insert_system(f"Failed to extract sufficient text from: {Path(path).name}")
                return

            doc_name = Path(path).name
            self.chat_insert_system(f"File '{doc_name}' added to indexing queue...")
            self.indexer.enqueue(text, doc_name)
            self.indexed_files.add(fid)

        except Exception as e:
            messagebox.showerror("Error", str(e))
        finally:
            self.set_busy(False)

    def on_send(self):
        if self.busy:
            return
        query = self.entry.get("1.0", "end").strip()
        if not query:
            return
        self.entry.delete("1.0", "end")
        self.chat_insert("YOU", query)

        self.set_busy(True, "Searching...")

        doc_filter = None
        for fname in self.indexed_files:
            if fname.lower() in query.lower():
                doc_filter = fname
                break

        docs = self.vs.search(query, k=CONFIG.TOP_K, doc_filter=doc_filter)
        history_context = self.memory.get_recent_history()
        answer = self.qa.answer(docs, query, history_context)

        self.chat_insert("ASSISTANT", answer)
        self.memory.add("USER", query)
        self.memory.add("ASSISTANT", answer)

        self.set_busy(False, "Ready")



# =============================================================================
# RUN APP
# =============================================================================

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()