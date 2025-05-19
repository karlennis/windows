import os
import re
import requests
import boto3
from dotenv import load_dotenv
from pinecone import Pinecone
from utils.config import PINECONE_API_KEY, OPENAI_API_KEY, S3_BUCKET, S3_REGION
import openai

load_dotenv()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("openai-docs")
s3 = boto3.client(
    's3',
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION")
)
openai.api_key = os.getenv("OPENAI_API_KEY")

def search_pinecone(query_text, top_k=50, filter_obj=None):
    SIMILARITY_THRESHOLD = 0.2
    FINAL_TOP_K = 15
    NAMESPACE = "default"
    embedding = openai.embeddings.create(model="text-embedding-3-small", input=[query_text])
    query_embedding = embedding.data[0].embedding
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        filter=filter_obj,
        namespace=NAMESPACE
    )
    if not results.get("matches"):
        return [], {}
    enriched = []
    for match in results["matches"]:
        metadata = match.get("metadata", {})
        project_id = str(metadata.get("project_id", match["id"].split("_chunk_")[0]))
        chunk_number = match["id"].split("_chunk_")[-1]
        score = match["score"]
        chunk_text_value = metadata.get("chunk_text") or fetch_chunk_from_s3(project_id, int(chunk_number))
        keyword_hits = sum(word.lower() in chunk_text_value.lower() for word in query_text.split())
        hybrid_score = score + 0.01 * keyword_hits
        enriched.append({
            "project_id": project_id,
            "chunk_number": chunk_number,
            "original_score": score,
            "hybrid_score": hybrid_score,
            "chunk_text": chunk_text_value
        })
    filtered = [m for m in enriched if m["original_score"] >= SIMILARITY_THRESHOLD]
    top_results = sorted(filtered, key=lambda x: x["hybrid_score"], reverse=True)[:FINAL_TOP_K]
    chunk_data = {}
    for m in top_results:
        chunk_data.setdefault(m["project_id"], []).append(m["chunk_text"])
    return list(chunk_data.keys()), chunk_data

def fetch_chunk_from_s3(application_number, chunk_index, chunk_size=300):
    s3_key = f"planning_documents_2025_04/{application_number}/docfiles.txt"
    try:
        response = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)
        full_text = response["Body"].read().decode("utf-8")
    except Exception as e:
        return ""
    words = full_text.split()
    start = chunk_index * chunk_size
    end = start + chunk_size
    return " ".join(words[start:end])

def fetch_full_text(project_id):
    s3_key = f"planning_documents_2025_04/{project_id}/docfiles.txt"
    try:
        response = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)
        return response["Body"].read().decode("utf-8")
    except Exception:
        return ""

def fetch_contact_details(project_ids):
    contact_query = ("Find the 'Applicant Details' or contact section including names, emails, phone numbers, and company details. ")
    filter_obj = {"project_id": {"$in": list(project_ids)}}
    _, contact_chunk_data = search_pinecone(contact_query, top_k=7 * len(project_ids), filter_obj=filter_obj)
    return {app: "\n".join(parts) for app, parts in contact_chunk_data.items()}

def fetch_feature_chunks(project_ids, feature):
    feature_docs = {}
    for project_id in project_ids:
        full_text = fetch_full_text(project_id)
        if full_text:
            sentences = re.split(r'(?<=[.!?])\s+', full_text)
            matching_sentences = [s for s in sentences if feature.lower() in s.lower()]
            feature_docs[project_id] = " ".join(matching_sentences) if matching_sentences else "No mentions found."
        else:
            feature_docs[project_id] = "No document available."
    return feature_docs

def extract_project_details(text):
    extract_prompt = (
        "Extract structured project details from the following applicant info. "
        "Include: Project Title, Project Description (2-3 sentences), Project Location, "
        "Contact Person Names, Phone Numbers, Emails, and Organization names. Text: " + text
    )
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": extract_prompt}]
    )
    return response.choices[0].message.content

def extract_feature_mentions(question, project_text):
    prompt = f"""
You are an expert in interpreting planning application documents. Your task is to analyze the provided text for any direct or indirect mentions of the following topic:

**Query:** "{question}"

Perform the following:
1. Identify and list any sentences that refer to the query. Be specific — include full sentences.
2. Provide a clear, concise summary (2-3 sentences) of the relevant information, focusing on what would matter most to a supplier (e.g., scope, timing, materials, scale, responsible parties, or functional requirements).
3. If the feature is **not mentioned or relevant**, state clearly that there are no mentions or implications.

Use professional, insight-driven language that would help suppliers assess opportunities or requirements quickly.

**Planning Application Text:**
{project_text}
"""

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


def generate_answer(question, chunk_data, api_details=None):
    if not chunk_data:
        return "No relevant documents found."
    combined_context = "\n\n".join(["\n".join(chunks) for chunks in chunk_data.values()])
    header = ""
    if api_details:
        headers = []
        for pid, details in api_details.items():
            headers.append(
                f"**Project {pid} - {details['planning_title']}**\n"
                f"Last researched on {details['planning_public_updated']}. Stage: {details['planning_stage']}\n"
                f"{details['planning_urlopen']}"
            )
        header = "\n\n".join(headers) + "\n\n"
    prompt = f"{header} You are an expert in planning applications. Answer the following question: \"{question}\"\n\nDocuments:\n{combined_context}"
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt.strip()}]
    )
    return response.choices[0].message.content

def get_one_thousand(limit_start: int, params_object: dict) -> str:
    base_url = os.getenv("BUILDING_INFO_API_BASE_URL", "https://api12.buildinginfo.com/api/v2/bi/projects/t-projects")
    api_key = os.getenv("BUILDING_INFO_API_KEY")
    ukey = os.getenv("BUILDING_INFO_API_UKEY")
    api_url = f"{base_url}?api_key={api_key}&ukey={ukey}"
    if params_object.get("category") not in (None, 0):
        api_url += f"&category={params_object['category']}"
    if params_object.get("subcategory") not in (None, 0):
        api_url += f"&subcategory={params_object['subcategory']}"
    if params_object.get("county") not in (None, 0):
        api_url += f"&county={params_object['county']}"
    if params_object.get("type") not in (None, 0):
        api_url += f"&type={params_object['type']}"
    if params_object.get("stage") not in (None, 0):
        api_url += f"&stage={params_object['stage']}"
    if params_object.get("latitude") and params_object.get("longitude") and params_object.get("radius"):
        api_url += f"&nearby={params_object['latitude']},{params_object['longitude']}&radius={params_object['radius']}"
    api_url += "&_apion=1.1"
    api_url += f"&more=limit {limit_start},1000"
    return api_url

def get_projects_by_params(params_object: dict):
    limit_start = 0
    all_project_ids = []
    all_rows = []
    while True:
        api_url = get_one_thousand(limit_start, params_object)
        try:
            response = requests.get(api_url)
            response.raise_for_status()
            data = response.json()
        except Exception:
            break
        if data.get("status") != "OK" or "data" not in data:
            break
        rows = data["data"].get("rows", [])
        if not rows:
            break
        project_ids = [row.get("planning_id") for row in rows if row.get("planning_id")]
        all_project_ids.extend(project_ids)
        all_rows.extend(rows)
        if len(rows) < 1000:
            break
        limit_start += 1000
    return all_project_ids, all_rows

def query_pipeline(query_payload):
    if isinstance(query_payload, dict):
        original_query = query_payload.get("search_query", "").strip()
        api_params = query_payload.get("api_params", {})
        is_report = query_payload.get("report", False)
    else:
        original_query = query_payload.strip()
        api_params = {}
        is_report = original_query.lower().startswith("report:")
    query_term = original_query
    if is_report and query_term.lower().startswith("report:"):
        query_term = query_term[len("report:"):].strip()

    if api_params:
        api_project_ids, api_data = get_projects_by_params(api_params)
    else:
        api_project_ids, api_data = [], []

    api_details = {}
    for row in api_data:
        pid = row.get("planning_id")
        if pid:
            api_details[str(pid)] = {
                "planning_title": row.get("planning_title", "N/A"),
                "planning_public_updated": row.get("planning_public_updated", "N/A"),
                "planning_stage": row.get("planning_stage", "N/A"),
                "planning_urlopen": row.get("planning_urlopen", "N/A")
            }

    filter_obj = {"project_id": {"$in": list(map(str, api_project_ids))}} if api_project_ids else None
    project_ids, chunk_data = search_pinecone(query_term, top_k=50, filter_obj=filter_obj)

    if api_project_ids:
        allowed_ids = set(str(pid) for pid in api_project_ids)
        project_ids = [pid for pid in project_ids if pid in allowed_ids]
        chunk_data = {pid: chunks for pid, chunks in chunk_data.items() if pid in allowed_ids}

    if not project_ids:
        return {"error": "No relevant projects found."}

    match_count = len(api_data) if api_params else 0

    if is_report:
        contact_chunks = fetch_contact_details(project_ids)
        feature_docs = fetch_feature_chunks(project_ids, query_term)
        report_projects = []
        for project_id in project_ids:
            details = {}
            details["project_id"] = project_id
            if project_id in api_details:
                meta = api_details[project_id]
                details.update(meta)
            else:
                details["planning_title"] = "N/A"
                details["planning_public_updated"] = "N/A"
                details["planning_stage"] = "N/A"
                details["planning_urlopen"] = "N/A"
            applicant_text = contact_chunks.get(project_id, "No applicant details found.")
            details["applicant_details"] = extract_project_details(applicant_text)
            feature_text = feature_docs.get(project_id, "No mentions found.")
            details["feature_mentions"] = extract_feature_mentions(query_term, feature_text)
            report_projects.append(details)

        summary_text = f"Report generated for query '{query_term}'. Found {len(report_projects)} project(s), with {match_count} matched documents via API."
        return {
            "search_query": query_term,
            "is_report": True,
            "match_count": match_count,
            "projects": report_projects,
            "response": summary_text
        }
    else:
        answer = generate_answer(query_term, chunk_data, api_details)
        return {
            "search_query": query_term,
            "is_report": False,
            "response": answer
        }
