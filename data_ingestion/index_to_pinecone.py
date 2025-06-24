import os
import sys
import argparse
import time
import threading
import traceback

import boto3
import openai
import tiktoken
from dotenv import load_dotenv
from nltk.tokenize import sent_tokenize
from pinecone import Pinecone
from concurrent.futures import ThreadPoolExecutor, as_completed

# allow imports from ../utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import S3_BUCKET, S3_REGION

# ─── Configuration ─────────────────────────────────────────────────────────────
load_dotenv()
openai.api_key   = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME       = "openai-docs"

# Chunking parameters
MAX_WORDS_PER_CHUNK = 300
OVERLAP_WORDS       = 50

# Embedding + batching parameters
MAX_MODEL_TOKENS    = 8191      # model limit minus 1 token
UPSERT_BYTES_LIMIT  = 2 * 1024 * 1024  # 2 MB
MAX_RETRIES         = 5
BACKOFF_FACTOR      = 2         # backoff multiplier

# Concurrency
MAX_WORKERS_DEFAULT = max(1, (os.cpu_count() or 1) // 2)

# ─── Clients & Globals ────────────────────────────────────────────────────────
pc    = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
s3    = boto3.client("s3", region_name=S3_REGION)

NAMESPACE = None
DIMENSION = pc.describe_index(name=INDEX_NAME)["dimension"]

# Token-bucket for TPM limits
class TokenBucket:
    def __init__(self, rate_per_sec: float, capacity: float):
        self.rate = rate_per_sec
        self.capacity = capacity
        self.tokens = capacity
        self.timestamp = time.time()
        self.lock = threading.Lock()
    def consume(self, amount: float):
        with self.lock:
            now = time.time()
            delta = now - self.timestamp
            self.tokens = min(self.capacity, self.tokens + delta * self.rate)
            self.timestamp = now
            if self.tokens >= amount:
                self.tokens -= amount
                return
            needed = amount - self.tokens
            time.sleep(needed / self.rate)
            self.tokens = 0
            self.timestamp = time.time()

bucket  = TokenBucket(rate_per_sec=1000000/60, capacity=1000000/60)
encoder = tiktoken.encoding_for_model("text-embedding-3-small")


# ─── Helpers ───────────────────────────────────────────────────────────────────
def chunk_text(text: str) -> list[str]:
    sentences = sent_tokenize(text)
    chunks, curr, curr_count = [], [], 0
    for sent in sentences:
        words = sent.split()
        if len(words) > MAX_WORDS_PER_CHUNK:
            if curr:
                chunks.append(" ".join(curr)); curr, curr_count = [], 0
            for i in range(0, len(words), MAX_WORDS_PER_CHUNK):
                chunks.append(" ".join(words[i:i+MAX_WORDS_PER_CHUNK]))
            continue
        if curr_count + len(words) > MAX_WORDS_PER_CHUNK:
            chunks.append(" ".join(curr))
            overlap = curr[-OVERLAP_WORDS:] if len(curr) > OVERLAP_WORDS else curr
            curr, curr_count = overlap.copy(), len(overlap)
        curr.extend(words); curr_count += len(words)
    if curr:
        chunks.append(" ".join(curr))
    return chunks

def split_by_token_limit(chunks: list[str]) -> list[list[str]]:
    """
    Split list of text chunks into sub-batches,
    each <= MAX_MODEL_TOKENS when summed.
    """
    batches, batch, count = [], [], 0
    for txt in chunks:
        toks = len(encoder.encode(txt))
        if toks > MAX_MODEL_TOKENS:
            # split this chunk itself by tokens
            words = txt.split()
            sub, i = [], 0
            while i < len(words):
                piece = []
                piece_count = 0
                while i < len(words):
                    w = words[i]
                    wc = len(encoder.encode(w + " "))
                    if piece_count + wc > MAX_MODEL_TOKENS:
                        break
                    piece.append(w)
                    piece_count += wc
                    i += 1
                batches.append([" ".join(piece)])
            continue
        if count + toks > MAX_MODEL_TOKENS:
            batches.append(batch)
            batch, count = [], 0
        batch.append(txt)
        count += toks
    if batch:
        batches.append(batch)
    return batches

def embed_text(chunks: list[str]) -> list[list[float]]:
    embeddings = []
    sub_batches = split_by_token_limit(chunks)
    for idx, batch in enumerate(sub_batches):
        bucket.consume(sum(len(encoder.encode(t)) for t in batch))
        for attempt in range(1, MAX_RETRIES+1):
            try:
                resp = openai.embeddings.create(
                    model="text-embedding-3-small",
                    input=batch
                )
                embeddings.extend([d.embedding for d in resp.data])
                break
            except Exception:
                traceback.print_exc()
                time.sleep(BACKOFF_FACTOR ** (attempt-1))
        else:
            print(f"❌ Failed to embed sub-batch {idx} after {MAX_RETRIES} attempts")
    return embeddings

def upsert_vectors(vectors: list[dict]):
    import json
    batch, size = [], 0
    for vec in vectors:
        vj = json.dumps(vec, separators=(",",":")).encode()
        vsz = len(vj)
        if size + vsz > UPSERT_BYTES_LIMIT:
            index.upsert(vectors=batch, namespace=NAMESPACE)
            batch, size = [], 0
        batch.append(vec)
        size += vsz
    if batch:
        index.upsert(vectors=batch, namespace=NAMESPACE)

def already_indexed(pid: str) -> bool:
    try:
        zero = [0.0]*DIMENSION
        res = index.query(vector=zero, top_k=1, namespace=NAMESPACE, filter={"project_id":pid})
        return bool(res.matches)
    except:
        return False

def process_and_index_document(pid: str, prefix: str):
    if already_indexed(pid):
        return
    key = f"{prefix}/{pid}/docfiles.txt"
    try:
        text = s3.get_object(Bucket=S3_BUCKET, Key=key)["Body"].read().decode()
    except Exception as e:
        print(f"S3 fetch failed for {pid}: {e}"); return
    chunks = chunk_text(text)
    print(f"{pid} → {len(chunks)} chunks")
    embs = embed_text(chunks)
    vecs = [
        {"id":f"{pid}_chunk_{i}", "values":vec, "metadata":{"project_id":pid}}
        for i, vec in enumerate(embs)
    ]
    upsert_vectors(vecs)

def index_all_projects(prefix: str, workers: int):
    pages = s3.get_paginator("list_objects_v2").paginate(
        Bucket=S3_BUCKET, Prefix=f"{prefix}/", Delimiter="/"
    )
    pids = [cp["Prefix"].rstrip("/").split("/")[-1]
            for pg in pages for cp in pg.get("CommonPrefixes",[])]
    print(f"Found {len(pids)} projects")
    with ThreadPoolExecutor(max_workers=workers) as exe:
        futures = [exe.submit(process_and_index_document, pid, prefix) for pid in pids]
        for f in as_completed(futures):
            f.result()

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("s3_prefix")
    parser.add_argument("-w","--workers",type=int,default=MAX_WORKERS_DEFAULT)
    args = parser.parse_args()

    NAMESPACE = args.s3_prefix
    if input(f"Clear namespace '{NAMESPACE}'? (yes)> ").lower()=="yes":
        index.delete(delete_all=True, namespace=NAMESPACE)
        print("Cleared.")
    index_all_projects(args.s3_prefix, args.workers)
