import os
import re
import json
import boto3
import requests
from dotenv import load_dotenv
from pinecone import Pinecone
import openai

# ─── Environment & Clients ────────────────────────────────────────────────────
load_dotenv()

# Pinecone setup
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("openai-docs")
NAMESPACE = "visqueen"

# S3 setup
s3 = boto3.client(
    's3',
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION")
)

# OpenAI setup
openai.api_key = os.getenv("OPENAI_API_KEY")


# ─── Pinecone + S3 Helpers ────────────────────────────────────────────────────
def fetch_chunk_from_s3(project_id: str, chunk_index: int, chunk_size: int = 300) -> str:
    """
    Fetch a single chunk from the full document stored in S3.
    """
    key = f"visqueen/{project_id}/docfiles.txt"
    try:
        resp = s3.get_object(Bucket=os.getenv("S3_BUCKET"), Key=key)
        full_text = resp['Body'].read().decode('utf-8')
    except Exception as e:
        print(f"⚠️ S3 fetch failed for {project_id}: {e}")
        return ""
    words = full_text.split()
    start = chunk_index * chunk_size
    return " ".join(words[start:start + chunk_size])


def fetch_full_text(project_id: str) -> str:
    """
    Fetch the full planning document text from S3.
    """
    key = f"visqueen/{project_id}/docfiles.txt"
    try:
        resp = s3.get_object(Bucket=os.getenv("S3_BUCKET"), Key=key)
        return resp['Body'].read().decode('utf-8')
    except Exception as e:
        print(f"⚠️ Full-text fetch failed for {project_id}: {e}")
        return ""


def search_pinecone(query_text: str,
                    top_k: int = 50,
                    filter_obj: dict = None):
    """
    Semantic + keyword hybrid search over Pinecone index.
    Returns: (project_id_list, chunk_data dict)
    """
    SIM_THRESHOLD = 0.2
    FINAL_TOP_K = 15

    # 1) Embed
    emb = openai.embeddings.create(
        model="text-embedding-3-small",
        input=[query_text]
    )
    query_emb = emb.data[0].embedding

    # 2) Query Pinecone
    resp = index.query(
        vector=query_emb,
        top_k=top_k,
        include_metadata=True,
        filter=filter_obj or {},
        namespace=NAMESPACE
    )
    matches = resp.get("matches", [])
    if not matches:
        return [], {}

    # 3) Hybrid score & collect
    enriched = []
    for m in matches:
        md = m.get("metadata", {})
        pid = str(md.get("project_id", m["id"].split("_chunk_")[0]))
        chunk_no = int(m["id"].split("_chunk_")[-1])
        score = m["score"]
        chunk_text = md.get("chunk_text") or fetch_chunk_from_s3(pid, chunk_no)
        keyword_hits = sum(1 for w in query_text.split() if w.lower() in chunk_text.lower())
        hybrid_score = score + 0.01 * keyword_hits
        enriched.append({
            "project_id": pid,
            "chunk_number": chunk_no,
            "original_score": score,
            "hybrid_score": hybrid_score,
            "chunk_text": chunk_text
        })

    # 4) Threshold & top-K
    filtered = [e for e in enriched if e["original_score"] >= SIM_THRESHOLD]
    topn = sorted(filtered, key=lambda x: x["hybrid_score"], reverse=True)[:FINAL_TOP_K]

    chunk_data: dict[str, list[str]] = {}
    project_ids: list[str] = []
    for e in topn:
        pid = e["project_id"]
        chunk_data.setdefault(pid, []).append(e["chunk_text"])
        if pid not in project_ids:
            project_ids.append(pid)

    return project_ids, chunk_data


# ─── API Pagination Helpers ───────────────────────────────────────────────────
def get_one_thousand(limit_start: int, params: dict) -> str:
    """
    Build URL to fetch 1,000 records from the Building Info API.
    """
    base = os.getenv("BUILDING_INFO_API_BASE_URL",
                     "https://api12.buildinginfo.com/api/v2/bi/projects/t-projects")
    key = os.getenv("BUILDING_INFO_API_KEY")
    ukey = os.getenv("BUILDING_INFO_API_UKEY")
    url = f"{base}?api_key={key}&ukey={ukey}&_apion=1.1&more=limit {limit_start},1000"

    for fld in ("category", "subcategory", "county", "type", "stage"):
        if params.get(fld):
            url += f"&{fld}={params[fld]}"

    if params.get("latitude") and params.get("longitude") and params.get("radius"):
        url += f"&nearby={params['latitude']},{params['longitude']}&radius={params['radius']}"

    print("GET URL:", url)
    return url


def get_projects_by_params(params: dict):
    """
    Repeatedly page through the API to collect all matching projects.
    Returns: (project_id_list, full_row_list)
    """
    start = 0
    all_ids = []
    all_rows = []
    while True:
        url = get_one_thousand(start, params)
        try:
            r = requests.get(url)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print("API error:", e)
            break

        if data.get("status") != "OK":
            break
        rows = data["data"].get("rows", [])
        if not rows:
            break

        batch_ids = [str(rw["planning_id"]) for rw in rows if rw.get("planning_id")]
        all_ids.extend(batch_ids)
        all_rows.extend(rows)
        if len(rows) < 1000:
            break
        start += 1000

    return all_ids, all_rows


# ─── OpenAI–Driven Feature Expansion & Extraction ─────────────────────────────
def expand_query_terms(brand: str) -> list[str]:
    """
    Ask OpenAI to return a JSON list of related generic feature keywords for a brand.
    """
    prompt = f"""
You are an expert in construction materials.
Given the brand name "{brand}", return up to 5 related generic product/feature keywords
(e.g. "paving", "block paving", "permeable paving") as a JSON array of strings.
If unknown, return [].
"""
    resp = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt.strip()}],
        temperature=0
    )
    txt = resp.choices[0].message.content.strip()
    try:
        arr = json.loads(txt)
        if isinstance(arr, list) and all(isinstance(x, str) for x in arr):
            return arr
    except json.JSONDecodeError:
        pass
    return []


def fetch_contact_details(project_ids: list[str]) -> dict[str, str]:
    """
    Semantic search to pull Applicant Details sections for each project.
    """
    query = "Find the 'Applicant Details' section with names, emails, phones, and organization."
    filt = {"project_id": {"$in": project_ids}}
    _, data = search_pinecone(query, top_k=7 * len(project_ids), filter_obj=filt)
    return {pid: "\n".join(chunks) for pid, chunks in data.items()}


def fetch_feature_chunks(project_ids: list[str], features: list[str]) -> dict[str,str]:
    """
    For each project, pull the full text and return only those
    sentences that mention ANY of the feature terms.
    """
    out: dict[str,str] = {}
    for pid in project_ids:
        full = fetch_full_text(pid)
        if not full:
            continue
        sentences = re.split(r'(?<=[.!?])\s+', full)
        hits = [s for s in sentences if any(f.lower() in s.lower() for f in features)]
        if hits:
            out[pid] = " ".join(hits)
    return out


def extract_project_details(text: str) -> str:
    """
    Use OpenAI to structure the applicant details block.
    """
    prompt = f"""
You are an expert in planning applications. Extract:
- Project Title
- Description (2-3 sentences)
- Location
- Contact Person(s)
- Phone Number(s)
- Email(s)
- Organization(s)

From this text:
{text}
"""
    resp = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt.strip()}],
        temperature=0
    )
    return resp.choices[0].message.content.strip()


def extract_feature_mentions(question: str, project_text: str) -> str:
    """
    Use OpenAI to list sentences that mention the question/features and summarize.
    """
    prompt = f"""
You are an expert in planning applications. From the text below,
list any sentences that mention "{question}" and give a 2-3 sentence summary.
If none, say "No mentions found."

Text:
{project_text}
"""
    resp = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt.strip()}],
        temperature=0
    )
    return resp.choices[0].message.content.strip()


# ─── Report & Answer Generation ────────────────────────────────────────────────
def generate_report(
    question: str,
    project_ids: list[str],
    chunk_data: dict[str,list[str]],
    api_details: dict[str,dict] | None = None
) -> str:
    # 1) expand brand→features via OpenAI (you already have this)
    synonyms = expand_query_terms(question)
    features = synonyms or [question]

    # 2) pull out only those projects that actually mention the feature
    feature_docs = fetch_feature_chunks(project_ids, features)
    # *** this is the critical filter ***
    valid_pids = list(feature_docs.keys())
    if not valid_pids:
        return "No relevant projects found."

    # 3) now fetch applicant/contact details for just those valid ones
    contact_docs = fetch_contact_details(valid_pids)

    sections = []
    for pid in valid_pids:
        proj_info     = extract_project_details(contact_docs.get(pid, ""))   # Applicant block
        mentions_text = extract_feature_mentions(question, feature_docs[pid]) # Summaries

        # Build header only if API metadata is in play
        header = ""
        if api_details and pid in api_details:
            d = api_details[pid]
            header = (
                f"**Project {pid} – {d['planning_title']}**\n"
                f"Last researched on {d['planning_public_updated']}. Stage: {d['planning_stage']}\n"
                f"{d['planning_urlopen']}\n\n"
            )

        sections.append(
            f"{header}"
            f"**Project {pid} Details:**\n{proj_info}\n\n"
            f"**Mentions of “{question}”:**\n{mentions_text}\n"
        )

    return "\n\n".join(sections)


def generate_answer(
    question: str,
    chunk_data: dict[str, list[str]],
    api_details: dict[str, dict] | None = None
) -> str:
    """
    Fall-back: answer the question using combined chunk excerpts.
    """
    if not chunk_data:
        return "No relevant documents found."

    combined = "\n\n".join(["\n".join(chunks) for chunks in chunk_data.values()])

    header = ""
    if api_details:
        lines = []
        for pid, d in api_details.items():
            lines.append(
                f"**Project {pid} – {d['planning_title']}**\n"
                f"Last researched on {d['planning_public_updated']}. Stage: {d['planning_stage']}\n"
                f"{d['planning_urlopen']}"
            )
        header = "\n\n".join(lines) + "\n\n"

    prompt = f"""
{header}
You are an expert in planning applications. Answer the question:
"{question}"

Documents:
{combined}
"""
    resp = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt.strip()}],
        temperature=0
    )
    return resp.choices[0].message.content.strip()


# ─── Main Interactive Loop ────────────────────────────────────────────────────
def main():
    print("\n📜 Planning Applications Query System")
    print("Prefix with 'report:' to generate a detailed report.\n")

    while True:
        query = input("🔎 Enter your query: ").strip()
        if not query:
            continue
        if query.lower() == "exit":
            break

        use_api = input("Apply API filtering? (y/n): ").lower().startswith('y')
        api_details = {}
        filter_obj = None

        if use_api:
            params_input = input("Enter API params JSON (or leave blank): ").strip()
            try:
                params = json.loads(params_input) if params_input else {}
            except:
                print("⚠️ Invalid JSON, using no filters.")
                params = {}

            ids, rows = get_projects_by_params(params)
            if not ids:
                print("❌ No projects from API.")
                continue

            filter_obj = {"project_id": {"$in": ids}}
            for r in rows:
                pid = str(r.get("planning_id"))
                api_details[pid] = {
                    "planning_title": r.get("planning_title", ""),
                    "planning_public_updated": r.get("planning_public_updated", ""),
                    "planning_stage": r.get("planning_stage", ""),
                    "planning_urlopen": r.get("planning_urlopen", "")
                }

        # 1) semantic search
        project_ids, chunk_data = search_pinecone(query, top_k=10, filter_obj=filter_obj) \
            if use_api else search_pinecone(query)

        if not project_ids:
            print("❌ No relevant projects found.")
            continue

        # 2) report vs answer
        if query.lower().startswith("report:"):
            actual = query[len("report:"):].strip()
            report = generate_report(actual, project_ids, chunk_data,
                                     api_details if use_api else None)
            print("\n📝 Report:\n")
            print(report)
        else:
            answer = generate_answer(query, chunk_data,
                                     api_details if use_api else None)
            print("\n🧠 Answer:\n")
            print(answer)

        print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()
