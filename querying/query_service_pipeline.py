import os
import json
import re
import requests
import boto3
from dotenv import load_dotenv
from pinecone import Pinecone
from utils.config import PINECONE_API_KEY, OPENAI_API_KEY, S3_BUCKET, S3_REGION
import openai

# Load environment variables
load_dotenv()

# Initialize Pinecone, S3, and OpenAI
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("openai-docs")  # Use your appropriate Pinecone index name
s3 = boto3.client(
    's3',
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION")
)
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- Semantic Search and Document Fetch Functions ---


def search_pinecone(query_text, top_k=50, filter_obj=None):
    """
    Performs semantic search on the Pinecone index using the OpenAI embedding model.
    Returns a list of project IDs and a dictionary mapping each project ID to its document chunks.
    """
    SIMILARITY_THRESHOLD = 0.2
    FINAL_TOP_K = 15
    NAMESPACE = "default"

    # 1) Embed the query text using OpenAI embeddings
    embedding = openai.embeddings.create(
        model="text-embedding-3-small",
        input=[query_text]
    )
    query_embedding = embedding.data[0].embedding

    # 2) Query Pinecone
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        filter=filter_obj,
        namespace=NAMESPACE
    )

    if not results.get("matches"):
        print("⚠️ No relevant chunks found.")
        return [], {}

    # 3) Hybrid scoring
    enriched = []
    for match in results["matches"]:
        metadata = match.get("metadata", {})
        project_id = str(metadata.get("project_id", match["id"].split("_chunk_")[0]))
        chunk_number = match["id"].split("_chunk_")[-1]
        score = match["score"]

        # Fallback to S3 if chunk_text not in metadata
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

    # 4) Filter by threshold & take top
    filtered = [m for m in enriched if m["original_score"] >= SIMILARITY_THRESHOLD]
    top_results = sorted(filtered, key=lambda x: x["hybrid_score"], reverse=True)[:FINAL_TOP_K]

    chunk_data = {}
    for m in top_results:
        chunk_data.setdefault(m["project_id"], []).append(m["chunk_text"])
        print(f"- Project {m['project_id']} | Chunk {m['chunk_number']} | "
              f"Score: {m['original_score']:.2f} | Hybrid: {m['hybrid_score']:.2f}")

    return list(chunk_data.keys()), chunk_data


def fetch_chunk_from_s3(application_number, chunk_index, chunk_size=300):
    """
    Fetches a specific chunk from the full document stored in S3.
    """
    s3_key = f"planning_documents_2025_04/{application_number}/docfiles.txt"
    try:
        response = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)
        full_text = response["Body"].read().decode("utf-8")
    except Exception as e:
        print(f"⚠️ Failed to fetch {s3_key}: {e}")
        return ""
    words = full_text.split()
    start = chunk_index * chunk_size
    end = start + chunk_size
    return " ".join(words[start:end])


def fetch_full_text(project_id):
    """
    Fetches the full text of the document for the given project from S3.
    """
    s3_key = f"planning_documents_2025_04/{project_id}/docfiles.txt"
    try:
        response = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)
        return response["Body"].read().decode("utf-8")
    except Exception as e:
        print(f"⚠️ Failed to fetch full text for project {project_id}: {e}")
        return ""


# --- Contact and Feature Extraction Functions ---


def fetch_contact_details(project_ids):
    """
    Searches for 'Applicant Details' in the documents (using semantic search) to extract contact info.
    """
    contact_query = (
        "Find the 'Applicant Details' or contact section including names, emails, "
        "phone numbers, and company details."
    )
    filter_obj = {"project_id": {"$in": list(project_ids)}}
    _, contact_chunk_data = search_pinecone(contact_query, top_k=7 * len(project_ids), filter_obj=filter_obj)
    return {app: "\n".join(parts) for app, parts in contact_chunk_data.items()}


def fetch_feature_chunks(project_ids, feature):
    """
    For each project, fetches the full text from S3 and returns sentences that mention the given feature.
    """
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
    """
    Uses OpenAI to extract structured project details from a given text excerpt.
    """
    extract_prompt = f"""
    You are an expert in planning applications. Extract and structure the following details from the section labeled 'Applicant Details' or similar:
    - Project Title
    - Project Description (Summarized in 2-3 sentences)
    - Project Location
    - Contact Person Name(s)
    - Contact Phone Number(s)
    - Contact Email(s)
    - Associated Company or Organization names

    If details are missing, infer from context.

    Text:
    {text}
    """
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": extract_prompt.strip()}]
    )
    return response.choices[0].message.content


def extract_feature_mentions(question, project_text):
    """
    Uses OpenAI to analyze the text for mentions and specifications of the queried feature
    and return a summary.
    """
    feature_prompt = f"""
    You are an expert in planning applications. Analyze the following text for any mentions or specifications of '{question}'.
    If there are any sentences that mention '{question}', list them and provide a concise summary
    of the key details in 2-3 sentences. If no relevant details are found, simply state that there are no mentions.

    Text:
    {project_text}
    """
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": feature_prompt.strip()}]
    )
    return response.choices[0].message.content


def generate_answer(question, chunk_data, api_details=None):
    """
    Generates an AI-powered answer using combined document excerpts.
    If API metadata is available, it is included as a header.
    """
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
    prompt = f"""
    {header}
    You are an expert in planning applications. Consider the document excerpts carefully and answer the following question:
    "{question}"

    Documents:
    {combined_context}
    """
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt.strip()}]
    )
    return response.choices[0].message.content


# --- API Search Helpers ---


def get_one_thousand(limit_start: int, params_object: dict) -> str:
    """
    Construct the API URL to retrieve 1000 records starting at limit_start.
    Only append numeric parameters if they're not None or 0.
    """
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

    print("GET ONE THOUSAND URL:", api_url)
    return api_url


def get_projects_by_params(params_object: dict):
    """
    Calls the Building Information API using structured parameters.
    Returns a tuple containing the list of project IDs and the raw project data.
    """
    limit_start = 0
    all_project_ids = []
    all_rows = []
    while True:
        api_url = get_one_thousand(limit_start, params_object)
        print("Calling API URL:", api_url)

        try:
            response = requests.get(api_url)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print("API call failed:", e)
            break

        if data.get("status") != "OK" or "data" not in data:
            break

        rows = data["data"].get("rows", [])
        if not rows:
            break

        project_ids = [row.get("planning_id") for row in rows if row.get("planning_id")]
        print("Fetched", len(project_ids), "IDs from this batch.")
        all_project_ids.extend(project_ids)
        all_rows.extend(rows)

        if len(rows) < 1000:
            break
        limit_start += 1000

    print("Total project IDs fetched:", len(all_project_ids))
    return all_project_ids, all_rows


# --- The Query Pipeline Function ---


# Here's the updated `query_pipeline()` implementation to match the behavior of the original `query_service.py`.
# The main changes are:
# - Strip 'report:' from the actual search query and pass the cleaned query consistently.
# - Use `fetch_feature_chunks()` and `extract_feature_mentions()` exactly as in the original.
# - Respect `api_params` only when they are intentionally provided.
# - Apply the hybrid scoring semantic search with fallback to S3 chunks.

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

    # Step 1: Get API project IDs if any filters are applied
    if api_params:
        api_project_ids, api_data = get_projects_by_params(api_params)
    else:
        api_project_ids, api_data = [], []

    # Step 2: Build API details dictionary
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

    # Step 3: Use Pinecone to semantically search for matching projects
    filter_obj = {"project_id": {"$in": list(map(str, api_project_ids))}} if api_project_ids else None
    project_ids, chunk_data = search_pinecone(query_term, top_k=50, filter_obj=filter_obj)

    # Step 4: Filter results if we had API constraints
    if api_project_ids:
        allowed_ids = set(str(pid) for pid in api_project_ids)
        project_ids = [pid for pid in project_ids if pid in allowed_ids]
        chunk_data = {pid: chunks for pid, chunks in chunk_data.items() if pid in allowed_ids}

    if not project_ids:
        return {"error": "No relevant projects found."}

    if is_report:
        contact_chunks = fetch_contact_details(project_ids)
        feature_docs = fetch_feature_chunks(project_ids, query_term)

        report_sections = []
        for project_id in project_ids:
            header = ""
            if project_id in api_details:
                meta = api_details[project_id]
                header = (
                    f"**Project {project_id} - {meta['planning_title']}**\n"
                    f"Last researched on {meta['planning_public_updated']}. Stage: {meta['planning_stage']}\n"
                    f"{meta['planning_urlopen']}\n\n"
                )

            project_details = extract_project_details(
                contact_chunks.get(project_id, "No applicant details found.")
            )
            feature_text = feature_docs.get(project_id, "No mentions found.")
            feature_mentions = extract_feature_mentions(query_term, feature_text)

            report_sections.append(
                f"{header}"
                f"**Project {project_id} Details:**\n{project_details}\n\n"
                f"**Mentions of '{query_term}':**\n{feature_mentions}\n"
            )

        return {
            "search_query": query_term,
            "is_report": True,
            "response": "\n\n".join(report_sections)
        }

    else:
        answer = generate_answer(query_term, chunk_data, api_details)
        return {
            "search_query": query_term,
            "is_report": False,
            "response": answer
        }
