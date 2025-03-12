import boto3
import os
import json
import re
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from utils.config import PINECONE_API_KEY, OPENAI_API_KEY, S3_BUCKET, S3_REGION
import openai

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("planning-docs")
s3 = boto3.client('s3', region_name=S3_REGION)

openai.api_key = OPENAI_API_KEY

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_text(text):
    """Generate embeddings using local sentence-transformers model."""
    return embedding_model.encode(text, normalize_embeddings=True).tolist()

def search_pinecone(query_text, top_k=10):
    """Perform semantic search on Pinecone index to find relevant document chunks."""
    query_embedding = embed_text(query_text)
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=False)

    if not results.get("matches"):
        print("⚠️ No relevant chunks found.")
        return [], {}

    project_ids = set()
    chunk_data = {}

    print("\n🔍 **Top Relevant Chunks:**")
    for match in results['matches']:
        project_id, chunk_number = match['id'].split("_chunk_")
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
    s3_key = f"sample_projects_20250110/{application_number}/docfiles.txt"

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
    """Search for 'Applicant Details' or similar sections to extract contact information."""
    contact_query = "Find the 'Applicant Details' or contact section including names, emails, phone numbers, and company details."

    _, contact_chunk_data = search_pinecone(contact_query, top_k=3 * len(project_ids))

    combined_docs = {app: "\n".join(parts) for app, parts in contact_chunk_data.items()}
    return combined_docs

def extract_project_details(text):
    """Extract structured project details, ensuring OpenAI correctly identifies contacts."""
    extract_prompt = f"""
    You are an expert in planning applications. Extract and structure the following details from the section labeled 'Applicant Details' or any similar section:
    - Project Title (If missing, infer from context)
    - Project Description (Summarize in 2-3 sentences)
    - Project Location (If not stated, infer from document)
    - Contact Person Name(s) (List all available names)
    - Contact Phone Number(s) (Include all available numbers)
    - Contact Email(s) (Provide all listed emails)
    - Associated Company or Organization names (If missing, try to infer from context)

    Look specifically in sections labeled **'Applicant Details', 'Submitted By', 'Project Contact', 'Developer Contact'** and ensure no details are missed.

    If any of these details are unavailable, search the document for similar sections before saying 'Not provided.'

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

def generate_report(question, project_ids, chunk_data):
    """Generate a structured report including project details and feature mentions."""
    print("🔎 Fetching detailed project metadata...")
    contact_chunks = fetch_contact_details(project_ids)

    report_sections = []

    for project_id in project_ids:
        print(f"📊 Extracting details for project {project_id}...")

        # Extract applicant details
        project_details = extract_project_details(contact_chunks.get(project_id, "No applicant details found."))

        # Extract feature mentions
        feature_mentions = extract_feature_mentions(question, chunk_data.get(project_id, ["No mentions found."]))

        section = f"""
        **Project {project_id} Details:**
        {project_details}

        **Mentions of '{question}':**
        {feature_mentions}
        """
        report_sections.append(section)

    return "\n\n".join(report_sections)

def generate_answer(question, chunk_data):
    """Generate an AI-powered answer using the combined text of relevant chunks."""
    if not chunk_data:
        return "No relevant documents found."

    combined_context = "\n\n".join(["\n".join(chunks) for chunks in chunk_data.values()])

    prompt = f"""
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

def main():
    print("\n📜 Planning Applications Query System")
    print("Type your question below. To generate a report with contact details, prefix your query with 'report:'.")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("🔎 Enter your query: ")
        if query.lower() == 'exit':
            break

        if query.lower().startswith("report:"):
            actual_query = query[len("report:"):].strip()
            project_ids, chunk_data = search_pinecone(actual_query)

            if not project_ids:
                print("❌ No relevant projects found.")
                continue

            report = generate_report(actual_query, project_ids, chunk_data)
            print("\n📝 Generated Report:")
            print(report)
        else:
            project_ids, chunk_data = search_pinecone(query)
            if not chunk_data:
                print("❌ No relevant documents found.")
                continue

            answer = generate_answer(query, chunk_data)
            print("\n🧠 AI Response:")
            print(answer)

        print("\n" + "=" * 50)

if __name__ == "__main__":
    main()
