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

# Initialize Pinecone, OpenAI, and S3
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


def search_pinecone(query_text, top_k=10):
    """Perform semantic search on Pinecone index to find relevant document chunks."""
    query_embedding = embed_text(query_text)
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=False)

    if not results.get("matches"):
        return [], {}

    project_ids = set()
    chunk_data = {}

    for match in results['matches']:
        project_id, chunk_number = match['id'].split("_chunk_")
        project_ids.add(project_id)

        chunk_text = fetch_chunk_from_s3(project_id, int(chunk_number))
        if chunk_text:
            if project_id not in chunk_data:
                chunk_data[project_id] = []
            chunk_data[project_id].append(chunk_text)

    return list(project_ids), chunk_data


def fetch_chunk_from_s3(application_number, chunk_index, chunk_size=200):
    """Fetch only the relevant chunk from the full document in S3."""
    s3_key = f"planning_documents_2025_03/{application_number}/docfiles.txt"

    try:
        response = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)
        full_text = response["Body"].read().decode("utf-8")
    except Exception as e:
        return f"⚠️ Failed to fetch {s3_key}: {e}"

    words = full_text.split()
    start, end = chunk_index * chunk_size, (chunk_index + 1) * chunk_size
    return " ".join(words[start:end])


def extract_project_details(text):
    """Extract structured project details, ensuring OpenAI correctly identifies contacts."""
    extract_prompt = f"""
    You are an expert in planning applications. Extract and structure the following details from the section labeled 'Applicant Details' or similar:
    - Project Title
    - Project Description (Summarized)
    - Project Location
    - Contact Person Name(s)
    - Contact Phone Number(s)
    - Contact Email(s)
    - Associated Company or Organization names

    If any details are missing, try to infer from the document.

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

    Summarize key details in 2-3 sentences.
    """

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": feature_prompt.strip()}]
    )

    return response.choices[0].message.content


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


def query_pipeline(query_text):
    """
    Runs the full query pipeline:
    - Detects if it's a report request.
    - Searches Pinecone for relevant chunks.
    - Extracts applicant details and feature mentions if needed.
    - Generates an AI response or structured report.
    """
    is_report = query_text.lower().startswith("report:")
    cleaned_query = query_text[len("report:") :].strip() if is_report else query_text

    project_ids, chunk_data = search_pinecone(cleaned_query)

    if not project_ids:
        return {"error": "No relevant projects found."}

    if is_report:
        # Extract project details and mentions for a structured report
        report_sections = []
        for project_id in project_ids:
            project_details = extract_project_details(chunk_data.get(project_id, ["No details found."]))
            feature_mentions = extract_feature_mentions(cleaned_query, chunk_data.get(project_id, ["No mentions found."]))

            section = f"""
            **Project {project_id} Details:**
            {project_details}

            **Mentions of '{cleaned_query}':**
            {feature_mentions}
            """
            report_sections.append(section)

        return {
            "query": cleaned_query,
            "is_report": True,
            "response": "\n\n".join(report_sections),
        }

    else:
        # Generate a direct AI response
        return {
            "query": cleaned_query,
            "is_report": False,
            "response": generate_answer(cleaned_query, chunk_data),
        }
