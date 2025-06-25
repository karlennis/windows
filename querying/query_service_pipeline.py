"""
query_pipeline.py  –  June 2025 revision
• One embedding call  • One completion per project
• No `_apion=1.1` when searching (filter queries)
• Project summary contains only Title | Description | Address | Stage
"""

from __future__ import annotations

import logging
import os
import random
import re
import time
from typing import Any, Dict, List

import boto3
import openai
import requests
from dotenv import load_dotenv
from pinecone import Pinecone
from requests import HTTPError

# ───────────────────────────── Logging ────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────── Environment / external clients ───────────────────────
load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("openai-docs")

s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION"),
)

openai.api_key = os.getenv("OPENAI_API_KEY")

S3_BUCKET = os.getenv("S3_BUCKET")
NAMESPACE = os.getenv("PINECONE_NAMESPACE")

BI_BASE_URL = os.getenv(
    "BUILDING_INFO_API_BASE_URL",
    "https://api12.buildinginfo.com/api/v2/bi/projects/t-projects",
)
BI_API_KEY = os.getenv("BUILDING_INFO_API_KEY")
BI_UKEY = os.getenv("BUILDING_INFO_API_UKEY")

# ─────────────────────── OpenAI helpers (with back-off) ───────────────────────
def _chat_retry(messages, model="gpt-4o-mini", max_retries=4, base=0.7):
    for i in range(max_retries):
        try:
            r = openai.chat.completions.create(model=model, messages=messages)
            return r.choices[0].message.content
        except openai.error.RateLimitError:
            delay = base * (2 ** i) + random.uniform(0, base)
            log.warning("OpenAI 429 – retry in %.2fs (%d/%d)", delay, i + 1, max_retries)
            time.sleep(delay)
    raise RuntimeError("OpenAI rate-limit after retries")


def _embed_retry(text, model="text-embedding-3-small", max_retries=4, base=0.7):
    for i in range(max_retries):
        try:
            r = openai.embeddings.create(model=model, input=[text])
            return r.data[0].embedding
        except openai.error.RateLimitError:
            delay = base * (2 ** i) + random.uniform(0, base)
            log.warning("OpenAI embed 429 – retry in %.2fs", delay)
            time.sleep(delay)
    raise RuntimeError("OpenAI rate-limit after retries")

# ───────────────────────── Pinecone / S3 helpers ──────────────────────────────
def search_pinecone(query_text: str, filter_obj=None):


    SIMILARITY_THRESHOLD = 0.19       # keep legacy cut-off
    FINAL_TOP_K          = 15         # distinct projects
    CANDIDATE_TOP_K      = 250        # ask Pinecone for more chunks

    # 1 embed once
    vec = _embed_retry(query_text)

    # 2 query Pinecone for plenty of candidates
    res = index.query(
        vector=vec,
        top_k=CANDIDATE_TOP_K,
        include_metadata=True,
        filter=filter_obj,
        namespace=NAMESPACE,
    )
    if not res.get("matches"):
        return [], {}

    # 3 hybrid-score every chunk (dense score + keyword bonus)
    scored = []
    for m in res["matches"]:
        md  = m.get("metadata", {})
        pid = str(md.get("project_id", m["id"].split("_chunk_")[0]))
        chunk_no = int(m["id"].split("_chunk_")[-1])
        txt = md.get("chunk_text") or fetch_chunk_from_s3(pid, chunk_no)

        kw_hits = sum(w.lower() in txt.lower() for w in query_text.split())
        scored.append(
            dict(
                pid=pid,
                chunk=txt,
                orig=m["score"],
                hybrid=m["score"] + 0.01 * kw_hits,
            )
        )

    # 4 discard chunks whose *original* similarity is below the threshold
    scored = [s for s in scored if s["orig"] >= SIMILARITY_THRESHOLD]

    # 5 sort once by hybrid score
    scored.sort(key=lambda s: s["hybrid"], reverse=True)

    # 6 collect up to 15 *distinct* projects
    chunk_map: Dict[str, List[str]] = {}
    for s in scored:
        if len(chunk_map) == FINAL_TOP_K and s["pid"] not in chunk_map:
            break
        chunk_map.setdefault(s["pid"], []).append(s["chunk"])

    return list(chunk_map.keys()), chunk_map





def fetch_chunk_from_s3(pid: str, idx: int, size=300) -> str:
    key = f"{NAMESPACE}/{pid}/docfiles.txt"
    try:
        body = s3.get_object(Bucket=S3_BUCKET, Key=key)["Body"].read().decode("utf-8")
    except Exception:
        return ""
    words = body.split()
    return " ".join(words[idx * size : (idx + 1) * size])


def fetch_full_text(pid: str) -> str:
    key = f"{NAMESPACE}/{pid}/docfiles.txt"
    try:
        body = s3.get_object(Bucket=S3_BUCKET, Key=key)["Body"].read().decode("utf-8")
        return body
    except Exception:
        return ""

def bi_url(extra: str = "") -> str:
    """
    Build a BuildingInfo URL *without* the _apion=1.1 suffix
    so the API returns the full historical record.
    """
    return (
        f"{BI_BASE_URL}?api_key={BI_API_KEY}&ukey={BI_UKEY}{extra}"
    )



def call_bi(url: str) -> dict:
    log.info("BI  ▶  GET  %s", url)
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        log.info("BI  ◀  %s  OK", r.status_code)
        return r.json()
    except HTTPError as e:
        txt = getattr(e.response, "text", "")[:300]
        log.error("BI  HTTPError %s %s  %s", e.response.status_code, url, txt)
        raise
    except Exception:
        log.exception("BI  EXCEPTION %s", url)
        raise


def get_projects_by_params(params: dict):
    start, ids, rows = 0, [], []
    # NOTE: NO _apion for filter search
    while True:
        qs = "".join(f"&{k}={v}" for k, v in params.items() if v not in (None, 0))
        url = bi_url(qs + f"&more=limit {start},1000", add_apion=False)
        try:
            data = call_bi(url)
        except Exception:
            break
        batch = data.get("data", {}).get("rows", [])
        if not batch:
            break
        ids.extend(r.get("planning_id") for r in batch if r.get("planning_id"))
        rows.extend(batch)
        if len(batch) < 1000:
            break
        start += 1000
    return ids, rows


def fetch_project_row(pid: str) -> dict:
    try:
        data = call_bi(bi_url(f"&planning_id={pid}"))
        rows = data.get("data", {}).get("rows", [])
        if not rows:
            log.warning("BI  WARN project_id=%s  No rows returned", pid)
            return {"error": "No rows"}
        return rows[0]
    except Exception as e:
        log.warning("BI  WARN project_id=%s  %s", pid, e)
        return {"error": str(e)}

# ───────────────────── per-project feature extraction ─────────────────────────
def extract_feature_mentions(question: str, text: str) -> str:
    prompt = f"""
You are an expert in planning applications.
Identify any sentences mentioning **{question}** (directly or indirectly).
Quote them verbatim. Then give a concise 2-3 sentence summary for suppliers.
If nothing relevant, state "No mentions found".
Planning text:
{text[:6000]}""".strip()
    return _chat_retry([{"role": "user", "content": prompt}])

# ───────────────────── Utility formatting ─────────────────────────────────────
def summarise_row(r: dict) -> str:
    pairs = [
        ("planning_title", "Title"),
        ("planning_description", "Description"),
        ("planning_address", "Address"),
        ("planning_stage", "Stage"),
    ]
    return "\n".join(f"{lbl}: {r.get(k,'N/A')}" for k, lbl in pairs)


def build_feature_section(query: str, full_text: str) -> str:
    """
    Return the old-style feature-mention string.

    • If the literal query word/phrase appears, quote every sentence that
      contains it (max 12 sentences for brevity) and append a 2-3 sentence
      GPT summary.
    • Otherwise return the exact string 'No mentions found.' so the Angular
      filter can drop the project.
    """
    # --- literal match --------------------------------------------------------
    sentences = re.split(r'(?<=[.!?])\s+', full_text)
    hits      = [s for s in sentences if query.lower() in s.lower()]

    if not hits:
        return "No mentions found."

    quoted    = "\n".join(f'• {s.strip()}' for s in hits[:12])

    # --- short GPT summary (1 call) ------------------------------------------
    summary_prompt = (
        f"In 2–3 sentences, summarise why the text quoted above is relevant to "
        f'a supplier interested in **{query}** (scope, scale, timing, etc.).'
    )
    summary = _chat_retry(
        [{"role": "user", "content": summary_prompt}],
    )

    return f"{quoted}\n\n**Summary:** {summary.strip()}"

# ───────────────────────────── query_pipeline ────────────────────────────────
def query_pipeline(payload):
    # --- 1. Parse ----------------------------------------------------------------
    if isinstance(payload, dict):
        original = payload.get("search_query", "").strip()
        api_params = payload.get("api_params", {})
        is_report = payload.get("report", False)
    else:
        original = str(payload).strip()
        api_params, is_report = {}, original.lower().startswith("report:")

    query_term = original[len("report:"):].strip() if is_report and original.lower().startswith("report:") else original

    # --- 2. Optional API pre-filter ---------------------------------------------
    api_ids, api_rows = ([], [])
    if api_params:
        api_ids, api_rows = get_projects_by_params(api_params)

    api_details = {
        str(r["planning_id"]): {
            "planning_title": r.get("planning_title"),
            "planning_public_updated": r.get("planning_public_updated"),
            "planning_stage": r.get("planning_stage"),
            "planning_urlopen": r.get("planning_urlopen"),
        }
        for r in api_rows
        if r.get("planning_id")
    }

    # --- 3. Pinecone search ------------------------------------------------------
    filt = {"project_id": {"$in": [str(pid) for pid in api_ids]}} if api_ids else None
    project_ids, chunk_data = search_pinecone(query_term, filter_obj=filt)

    if api_ids:
        allow = {str(pid) for pid in api_ids}
        project_ids = [pid for pid in project_ids if pid in allow]
        chunk_data = {pid: txts for pid, txts in chunk_data.items() if pid in allow}

    if not project_ids:
        return {"error": "No relevant projects found."}

    # --- 4. REPORT mode ----------------------------------------------------------
    if is_report:
        match_count = len(api_rows)

        projects = []

        for pid in project_ids:
            row = fetch_project_row(pid)

            # NEW: use Pinecone chunks first; fall back to full text if none
            full_text         = fetch_full_text(pid)
            feature_mentions  = build_feature_section(query_term, full_text)
 # ★

            projects.append(
                {
                    "project_id": pid,
                    "planning_title": row.get("planning_title", "N/A"),
                    "planning_public_updated": row.get("planning_public_updated", "N/A"),
                    "planning_stage": row.get("planning_stage", "N/A"),
                    "planning_urlopen": row.get("planning_urlopen", "N/A"),
                    "project_summary": summarise_row(row),
                    "feature_mentions": feature_mentions,
                }
            )


        return {
            "search_query": query_term,
            "is_report": True,
            "match_count": match_count,
            "projects": projects,
            "response": (
                f"Report for '{query_term}'. {len(projects)} project(s) "
                f"({match_count} matched via API filters)."
            ),
        }

    # --- 5. ANSWER mode ----------------------------------------------------------
    context = "\n\n".join("\n".join(v) for v in chunk_data.values())
    header = ""
    if api_details:
        header = "\n\n".join(
            f"**Project {pid} – {d.get('planning_title','N/A')}**\n"
            f"Last updated: {d.get('planning_public_updated','N/A')}  Stage: {d.get('planning_stage','N/A')}\n"
            f"{d.get('planning_urlopen','')}"
            for pid, d in api_details.items()
        )

    answer = _chat_retry(
        [
            {
                "role": "user",
                "content": (
                    f"{header}\n\n"
                    f"Use the following planning-application fragments to answer:\n\n"
                    f"**Question:** \"{query_term}\"\n\n{context}"
                ),
            }
        ]
    )
    return {"search_query": query_term, "is_report": False, "response": answer}
