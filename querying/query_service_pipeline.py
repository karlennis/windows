"""
query_pipeline.py · June 2025
────────────────────────────────────────────────────────────────────────────
• Optional BuildingInfo parameter search → unlimited pages of 1 000 rows
  (always uses &_apion=1.1)
• If no API parameters: Pinecone search first, then one plain detail lookup
  *without* &_apion=1.1 for every planning_id
• Pinecone restricted to API IDs when filter is in use
• One embedding + one completion per project
• Dates reformatted to dd/MM/yyyy  → label “Updated on”
"""

from __future__ import annotations

import datetime as _dt
import logging
import os
import random
import re
import time
from typing import Dict, List

import boto3
import openai
import requests
from dotenv import load_dotenv
from pinecone import Pinecone
from requests import HTTPError

# ─────────────────────── Environment / clients ───────────────────────────────
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

# ───────────────────────────── Logging ───────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ═══════════════════════ OpenAI retry helpers ═════════════════════════════════
def _chat_retry(messages, model="gpt-4o-mini", max_retries=4, base=0.7):
    for i in range(max_retries):
        try:
            r = openai.chat.completions.create(model=model, messages=messages)
            return r.choices[0].message.content
        except openai.error.RateLimitError:
            delay = base * (2**i) + random.uniform(0, base)
            log.warning("OpenAI 429 – retry in %.2fs (%d/%d)", delay, i + 1, max_retries)
            time.sleep(delay)
    raise RuntimeError("OpenAI rate-limit after retries")


def _embed_retry(text, model="text-embedding-3-small", max_retries=4, base=0.7):
    for i in range(max_retries):
        try:
            r = openai.embeddings.create(model=model, input=[text])
            return r.data[0].embedding
        except openai.error.RateLimitError:
            delay = base * (2**i) + random.uniform(0, base)
            log.warning("OpenAI embed 429 – retry in %.2fs", delay)
            time.sleep(delay)
    raise RuntimeError("OpenAI rate-limit after retries")

# ═════════════════════ BuildingInfo helpers ═══════════════════════════════════
def _bi_call(url: str) -> dict:
    log.info("BI ▶ GET %s", url)
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        log.info("BI ◀ %s OK", resp.status_code)
        return resp.json()
    except HTTPError as e:
        log.error("BI HTTPError %s %s", e.response.status_code, url)
        raise
    except Exception:
        log.exception("BI EXCEPTION %s", url)
        raise


def _get_one_thousand(offset: int, params: dict) -> str:
    """
    Build a 1 000-row page URL with &_apion=1.1 (filter branch only).
    """
    url = f"{BI_BASE_URL}?api_key={BI_API_KEY}&ukey={BI_UKEY}"
    for key in ("category", "subcategory", "county", "type", "stage"):
        val = params.get(key)
        if val not in (None, 0, ""):
            url += f"&{key}={val}"
    if all(params.get(k) for k in ("latitude", "longitude", "radius")):
        url += (
            f"&nearby={params['latitude']},{params['longitude']}"
            f"&radius={params['radius']}"
        )
    url += "&_apion=1.1"
    url += f"&more=limit {offset},1000"
    return url


def get_projects_by_params(params: dict):
    offset, ids, rows = 0, [], []
    while True:
        url = _get_one_thousand(offset, params)
        try:
            data = _bi_call(url)
        except Exception:
            break
        batch = data.get("data", {}).get("rows", [])
        if not batch:
            break
        ids.extend(str(r["planning_id"]) for r in batch if r.get("planning_id"))
        rows.extend(batch)
        if len(batch) < 1000:
            break
        offset += 1000
    return ids, rows


def _detail_url(pid: str, *, add_apion: bool) -> str:
    """Single-row lookup; add &_apion=1.1 only when coming from a filter search."""
    suffix = "&_apion=1.1" if add_apion else ""
    return f"{BI_BASE_URL}?api_key={BI_API_KEY}&ukey={BI_UKEY}&planning_id={pid}{suffix}"


def fetch_project_row(pid: str, *, from_filter: bool) -> dict:
    try:
        data = _bi_call(_detail_url(pid, add_apion=from_filter))
        return data.get("data", {}).get("rows", [])[0]
    except Exception:
        return {}

# ═════════════════════ S3 helpers (chunks / full text) ════════════════════════
def fetch_chunk_from_s3(pid: str, idx: int, size=300) -> str:
    key = f"{NAMESPACE}/{pid}/docfiles.txt"
    try:
        txt = s3.get_object(Bucket=S3_BUCKET, Key=key)["Body"].read().decode("utf-8")
    except Exception:
        return ""
    words = txt.split()
    return " ".join(words[idx * size : (idx + 1) * size])


def fetch_full_text(pid: str) -> str:
    key = f"{NAMESPACE}/{pid}/docfiles.txt"
    try:
        return s3.get_object(Bucket=S3_BUCKET, Key=key)["Body"].read().decode("utf-8")
    except Exception:
        return ""

# ═════════════════════ Pinecone search (15 unique projects) ═══════════════════
def search_pinecone(query: str, *, filter_obj=None):
    SIM = 0.19
    TOP_CHUNKS = 250
    MAX_PROJ = 15

    vec = _embed_retry(query)
    res = index.query(
        vector=vec,
        top_k=TOP_CHUNKS,
        include_metadata=True,
        filter=filter_obj,
        namespace=NAMESPACE,
    )
    if not res.get("matches"):
        return [], {}

    scored = []
    for m in res["matches"]:
        md = m.get("metadata", {})
        pid = str(md.get("project_id", m["id"].split("_chunk_")[0]))
        idx = int(m["id"].split("_chunk_")[-1])
        txt = md.get("chunk_text") or fetch_chunk_from_s3(pid, idx)
        bonus = 0.01 * sum(w.lower() in txt.lower() for w in query.split())
        scored.append({"pid": pid, "chunk": txt, "sim": m["score"], "hyb": m["score"] + bonus})

    scored = [s for s in scored if s["sim"] >= SIM]
    scored.sort(key=lambda s: s["hyb"], reverse=True)

    chunks: Dict[str, List[str]] = {}
    for s in scored:
        if len(chunks) == MAX_PROJ and s["pid"] not in chunks:
            break
        chunks.setdefault(s["pid"], []).append(s["chunk"])

    return list(chunks.keys()), chunks

# ═════════════ Feature snippets + GPT summary helper ══════════════════════════
def build_feature_section(query: str, text: str, window=8, max_hits=20) -> str:
    sents = re.split(r"(?<=[.!?])\s+", text)
    hits = [s for s in sents if query.lower() in s.lower()]
    if not hits:
        return "No mentions found."

    snippets = []
    q_re = re.compile(re.escape(query), re.I)
    for s in hits[:max_hits]:
        m = q_re.search(s)
        if not m:
            continue
        start = m.start()
        words = s.split()
        char = 0
        w_idx = 0
        for i, w in enumerate(words):
            char += len(w) + 1
            if char > start:
                w_idx = i
                break
        lo = max(w_idx - window, 0)
        hi = min(w_idx + window + 1, len(words))
        snippets.append(f"• …{' '.join(words[lo:hi])}…")

    summary = _chat_retry(
        [
            {
                "role": "user",
                "content": (
                    f"In 2–3 sentences, summarise what suppliers need to know about "
                    f'**{query}** based on these snippets.'
                ),
            }
        ]
    )
    return "\n".join(snippets) + "\n\n**Summary:** " + summary.strip()

# ═════════════════════ query_pipeline (entry) ═════════════════════════════════
def query_pipeline(payload):
    # 1 Parse -------------------------------------------------------------------
    if isinstance(payload, dict):
        raw_query = payload.get("search_query", "").strip()
        api_params = payload.get("api_params", {})
        is_report = payload.get("report", False)
    else:
        raw_query = str(payload).strip()
        api_params, is_report = {}, raw_query.lower().startswith("report:")

    query_term = (
        raw_query[len("report:"):].strip()
        if is_report and raw_query.lower().startswith("report:")
        else raw_query
    )

    # 2 API filter (if any) -----------------------------------------------------
    api_ids, api_rows = [], []
    if api_params and any(v not in (None, 0, "") for v in api_params.values()):
        api_ids, api_rows = get_projects_by_params(api_params)

    api_meta = {
        r["planning_id"]: {
            "planning_title":        r.get("planning_title", "N/A"),
            "planning_description":  r.get("planning_description", "N/A"),   # ← NEW
            "updated_on": (
                _dt.datetime.fromisoformat(r["planning_public_updated"]).strftime("%d/%m/%Y")
                if r.get("planning_public_updated") else "N/A"
            ),
            "planning_stage":        r.get("planning_stage", "N/A"),
            "planning_urlopen":      r.get("planning_urlopen", "N/A"),
        }
        for r in api_rows
        if r.get("planning_id")
    }


    # 3 Pinecone ---------------------------------------------------------------
    filter_obj = {"project_id": {"$in": api_ids}} if api_ids else None
    pine_ids, chunk_map = search_pinecone(query_term, filter_obj=filter_obj)

    if api_ids:
        allow = set(api_ids)
        pine_ids = [pid for pid in pine_ids if pid in allow]
        chunk_map = {pid: txts for pid, txts in chunk_map.items() if pid in allow}

    if not pine_ids:
        return {"error": "No relevant projects found."}

    # 4 Report mode ------------------------------------------------------------
    if is_report:
        projects = []
        for pid in pine_ids:
            # meta from filter search, or fetch without _apion
            if int(pid) in api_meta:
                meta = api_meta[int(pid)]
            else:
                row = fetch_project_row(pid, from_filter=False)
                meta = {
                    "planning_title":        row.get("planning_title", "N/A"),
                    "planning_description":  row.get("planning_description", "N/A"),   # ← NEW
                    "updated_on": (
                        _dt.datetime.fromisoformat(row["planning_public_updated"]).strftime("%d/%m/%Y")
                        if row.get("planning_public_updated") else "N/A"
                    ),
                    "planning_stage":        row.get("planning_stage", "N/A"),
                    "planning_urlopen":      row.get("planning_urlopen", "N/A"),
                }

            projects.append(
                {
                    "project_id": pid,
                    **meta,
                    "feature_mentions": build_feature_section(
                        query_term, fetch_full_text(pid)
                    ),
                }
            )

        return {
            "search_query": query_term,
            "is_report": True,
            "match_count": len(api_rows),
            "projects": projects,
            "response": f"Report for '{query_term}' – {len(projects)} project(s).",
        }

    # 5 Answer mode ------------------------------------------------------------
    context = "\n\n".join("\n".join(v) for v in chunk_map.values())
    header = ""
    if api_meta:
        header = "\n\n".join(
            f"**Project {pid} – {m['planning_title']}**\n"
            f"Updated on {m['updated_on']}  Stage: {m['planning_stage']}\n"
            f"{m['planning_urlopen']}"
            for pid, m in api_meta.items()
        )

    answer = _chat_retry(
        [
            {
                "role": "user",
                "content": (
                    f"{header}\n\nUse the fragments below to answer "
                    f'the question **"{query_term}"**:\n\n{context}'
                ),
            }
        ]
    )
    return {"search_query": query_term, "is_report": False, "response": answer}
