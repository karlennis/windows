import boto3
import os
import json
import re
import requests
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from utils.config import PINECONE_API_KEY, OPENAI_API_KEY, S3_BUCKET, S3_REGION
import openai

load_dotenv()

# Initialize Pinecone, S3, and OpenAI
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("planning-docs")
s3 = boto3.client(
    's3',
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION")
)

openai.api_key = OPENAI_API_KEY
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


def embed_text(text):
    """Generate embeddings using local sentence-transformers model."""
    return embedding_model.encode(text, normalize_embeddings=True).tolist()


def search_pinecone(query_text, top_k=10, filter_obj=None):
    """
    Perform semantic search on Pinecone index.
    If a filter_obj is provided, it is passed along (though in our case, we
    rely on manual filtering since the vector IDs are formatted as "id_chunk_x").
    """
    query_embedding = embed_text(query_text)
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        filter=filter_obj,  # may be None
        include_metadata=True
    )

    if not results.get("matches"):
        print("⚠️ No relevant chunks found.")
        return [], {}

    project_ids = set()
    chunk_data = {}

    print("\n🔍 **Top Relevant Chunks:**")
    for match in results['matches']:
        # Try to use metadata if available; otherwise extract project id from the vector id
        project_id = match.get("metadata", {}).get("project_id", match["id"].split("_chunk_")[0])
        chunk_number = match["id"].split("_chunk_")[-1]
        project_ids.add(project_id)

        chunk_text = fetch_chunk_from_s3(project_id, int(chunk_number))
        if chunk_text:
            if project_id not in chunk_data:
                chunk_data[project_id] = []
            chunk_data[project_id].append(chunk_text)

        print(f"- Project ID: {project_id}, Chunk: {chunk_number}, Score: {match['score']:.2f}")

    return list(project_ids), chunk_data


def fetch_chunk_from_s3(application_number, chunk_index, chunk_size=200):
    """Fetch only the relevant chunk from the full document in S3."""
    s3_key = f"planning_documents_2025_03/{application_number}/docfiles.txt"

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


def fetch_contact_details(project_ids):
    """
    Search for 'Applicant Details' or similar sections to extract contact information.
    This function performs another Pinecone search using a contact-focused query.
    """
    contact_query = "Find the 'Applicant Details' or contact section including names, emails, phone numbers, and company details."
    # Multiply top_k by the number of projects to try to cover all relevant chunks.
    _, contact_chunk_data = search_pinecone(contact_query, top_k=3 * len(project_ids))
    combined_docs = {app: "\n".join(parts) for app, parts in contact_chunk_data.items()}
    return combined_docs


def extract_project_details(text):
    """Extract structured project details using OpenAI."""
    extract_prompt = f"""
    You are an expert in planning applications. Extract and structure the following details from the section labeled 'Applicant Details' or any similar section:
    - Project Title (If missing, infer from context)
    - Project Description (Summarize in 2-3 sentences)
    - Project Location (If not stated, infer from document)
    - Contact Person Name(s) (List all available names)
    - Contact Phone Number(s) (Include all available numbers)
    - Contact Email(s) (Provide all listed emails)
    - Associated Company or Organization names (If missing, try to infer from context)

    Look specifically in sections labeled **'Applicant Details', 'Submitted By', 'Project Contact', 'Developer Contact'**.

    Text:
    {text}
    """
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": extract_prompt.strip()}]
    )
    return response.choices[0].message.content


def extract_feature_mentions(question, project_chunks):
    """Extract mentions of the queried feature from the relevant chunks."""
    combined_text = "\n".join(project_chunks)
    feature_prompt = f"""
    Identify all mentions and specifications of '{question}' in the following planning document excerpts:
    {combined_text}

    Summarize the key details in 2-3 sentences.
    """
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": feature_prompt.strip()}]
    )
    return response.choices[0].message.content


def generate_report(question, project_ids, chunk_data, api_details=None):
    """Generate a structured report including API metadata, project details, and feature mentions."""
    print("🔎 Fetching detailed project metadata...")
    contact_chunks = fetch_contact_details(project_ids)
    report_sections = []

    for project_id in project_ids:
        header = ""
        if api_details and project_id in api_details:
            header = (
                f"**Project {project_id} - {api_details[project_id]['planning_title']}**\n"
                f"Last researched on {api_details[project_id]['planning_public_updated']}. Stage: {api_details[project_id]['planning_stage']}\n"
                f"{api_details[project_id]['planning_urlopen']}\n\n"
            )
        print(f"📊 Extracting details for project {project_id}...")
        project_details = extract_project_details(contact_chunks.get(project_id, "No applicant details found."))
        feature_mentions = extract_feature_mentions(question, chunk_data.get(project_id, ["No mentions found."]))
        section = f"{header}" + \
                  f"**Project {project_id} Details:**\n{project_details}\n\n" \
                  f"**Mentions of '{question}':**\n{feature_mentions}\n"
        report_sections.append(section)
    return "\n\n".join(report_sections)


def generate_answer(question, chunk_data, api_details=None):
    """Generate an AI-powered answer using combined document excerpts and API metadata if available."""
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
    You are an expert in planning applications. Using the following document excerpts, answer the question:
    "{question}"

    Documents:
    {combined_context}
    """
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt.strip()}]
    )
    return response.choices[0].message.content


# --- New API Searching Functions ---

def get_one_thousand(limit_start: int, params_object: dict) -> str:
    """
    Construct the API URL to retrieve 1000 records starting at limit_start.
    This follows the provided JavaScript example by adding parameters only if provided.
    """
    base_url = os.getenv("BUILDING_INFO_API_BASE_URL", "https://api12.buildinginfo.com/api/v2/bi/projects/t-projects")
    api_key = os.getenv("BUILDING_INFO_API_KEY")
    ukey = os.getenv("BUILDING_INFO_API_UKEY")

    # Initialize URL with API key and user key
    api_url = f"{base_url}?api_key={api_key}&ukey={ukey}"

    # Add category parameter if applicable
    if params_object.get("category") not in (0, None):
        api_url += "&category=" + str(params_object.get("category"))

    # Add subcategory parameter if applicable
    if params_object.get("subcategory") not in (0, None):
        api_url += "&subcategory=" + str(params_object.get("subcategory"))

    # Add county parameter if applicable
    if params_object.get("county") not in (0, None):
        api_url += "&county=" + str(params_object.get("county"))

    if params_object.get("type") not in (0, None):
        api_url += "&type=" + str(params_object.get("type"))

    if params_object.get("stage") not in (0, None):
        api_url += "&stage=" + str(params_object.get("stage"))

    # Add nearby and radius parameters if provided
    if params_object.get("latitude") and params_object.get("longitude") and params_object.get("radius"):
        api_url += f"&nearby={params_object['latitude']},{params_object['longitude']}&radius={params_object['radius']}"

    # Always add _apion=1 parameter (can be extended based on UI selections)
    api_url += "&_apion=1.1"

    # Add pagination parameter to retrieve 1000 records starting at limit_start
    api_url += f"&more=limit {limit_start},1000"

    print("GET ONE THOUSAND URL:", api_url)
    return api_url


def get_projects_by_params(params_object: dict):
    """
    Call the Building Information API using a structured parameter object.
    This retrieves all matching records by paginating through the results.
    Returns a tuple of (project_ids, full_project_data) aggregated over all pages.
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


def main():
    print("\n📜 Planning Applications Query System")
    print("Type your question below. To generate a report with contact details, prefix your query with 'report:'.")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("🔎 Enter your query: ")
        if query.lower() == 'exit':
            break

        # Ask if the user wants to apply API filtering via the new API search
        use_api = input("Apply API filtering? (y/n): ").lower().strip() == 'y'
        api_params = {}
        api_details = {}
        allowed_api_ids = []
        if use_api:
            print("Enter API parameters as a JSON string (e.g. {\"category\":1, \"county\":\"Dublin\"}). Leave empty for defaults:")
            params_input = input("🔧 API Parameters: ").strip()
            if params_input:
                try:
                    api_params = json.loads(params_input)
                except Exception as e:
                    print("Invalid JSON provided. Using empty parameters.")
                    api_params = {}

            # Get API projects (all pages)
            api_project_ids, api_data = get_projects_by_params(api_params)
            if not api_project_ids:
                print("❌ Docs don't exist.")
                print("\n" + "=" * 50 + "\n")
                continue

            allowed_api_ids = set(api_project_ids)

            # Build API details dictionary for each project
            for row in api_data:
                pid = row.get("planning_id")
                if pid:
                    api_details[pid] = {
                        "planning_title": row.get("planning_title", "N/A"),
                        "planning_public_updated": row.get("planning_public_updated", "N/A"),
                        "planning_stage": row.get("planning_stage", "N/A"),
                        "planning_urlopen": row.get("planning_urlopen", "N/A")
                    }

            # Instead of applying a Pinecone filter (which requires metadata),
            # we will manually filter the results after the search.
            project_ids, chunk_data = search_pinecone(query, top_k=10)
            filtered_project_ids = []
            filtered_chunk_data = {}
            for pid, chunks in chunk_data.items():
                if pid in allowed_api_ids:
                    filtered_project_ids.append(pid)
                    filtered_chunk_data[pid] = chunks
            project_ids = filtered_project_ids
            chunk_data = filtered_chunk_data

            # Print the matching project IDs from the API
            print("\n✅ Matching project IDs:", list(filtered_project_ids))
        else:
            project_ids, chunk_data = search_pinecone(query)

        if not project_ids:
            print("❌ No relevant projects found.")
            print("\n" + "=" * 50 + "\n")
            continue

        if query.lower().startswith("report:"):
            actual_query = query[len("report:"):].strip()
            report = generate_report(actual_query, project_ids, chunk_data, api_details)
            print("\n📝 Generated Report:")
            print(report)
        else:
            answer = generate_answer(query, chunk_data, api_details)
            print("\n🧠 AI Response:")
            print(answer)

        print("\n" + "=" * 50 + "\n")


if __name__ == "__main__":
    main()
