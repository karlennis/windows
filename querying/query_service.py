import boto3
import os
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

# Load free embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_text(text):
    """Generate embeddings using local sentence-transformers model."""
    return embedding_model.encode(text, normalize_embeddings=True).tolist()

def search_pinecone(query_text):
    query_embedding = embed_text(query_text)

    results = index.query(vector=query_embedding, top_k=10, include_metadata=False)

    if not results.get("matches"):
        print("⚠️ No relevant chunks found.")
        return []

    print("\n🔍 **Top Relevant Chunks:**")
    for match in results['matches']:
        print(f"- Chunk: {match['id']}, Score: {match['score']:.2f}")

    return [match['id'] for match in results['matches']]

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

def fetch_relevant_chunks(chunk_ids):
    documents = {}

    for chunk_id in chunk_ids:
        application_number, chunk_number = chunk_id.split("_chunk_")
        chunk_text = fetch_chunk_from_s3(application_number, int(chunk_number))
        if application_number not in documents:
            documents[application_number] = []
        documents[application_number].append(chunk_text)

    # Combine chunks per document
    combined_docs = {app: "\n".join(parts) for app, parts in documents.items()}
    return combined_docs

def generate_answer(question, documents):
    if not documents:
        return "No relevant documents found."

    combined_context = "\n\n".join([
        f"Document {app_number}:\n{text}"
        for app_number, text in documents.items()
    ])

    prompt = f"""
    You are an expert in planning applications. Using the following document excerpts, answer the question:
    "{question}"

    Documents:
    {combined_context}
    """

    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt.strip()}]
    )

    return response.choices[0].message.content

def main():
    print("\n📜 Planning Applications Query System (Chunked Retrieval - Local Embeddings)")
    print("Type your question below. Type 'exit' to quit.\n")

    while True:
        question = input("🔎 Ask a question: ")
        if question.lower() == 'exit':
            break

        chunk_ids = search_pinecone(question)

        if not chunk_ids:
            print("❌ No relevant documents found.")
            continue

        documents = fetch_relevant_chunks(chunk_ids)
        answer = generate_answer(question, documents)

        print("\n🧠 AI Response:")
        print(answer)
        print("\n" + "="*50)

if __name__ == "__main__":
    main()
