import boto3
from pinecone import Pinecone
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from utils.config import S3_BUCKET, S3_REGION, PINECONE_API_KEY

# Load environment variables
load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("planning-docs")

# Initialize S3
s3 = boto3.client('s3', region_name=S3_REGION)

# Initialize free embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_text(text):
    """Generate embeddings using local sentence-transformers model."""
    return embedding_model.encode(text, normalize_embeddings=True).tolist()

def chunk_text(text, max_words=200):
    """Split text into smaller chunks for embedding."""
    words = text.split()
    for i in range(0, len(words), max_words):
        yield " ".join(words[i:i + max_words])

def process_and_index_document(application_number):
    s3_key = f"sample_projects_20250110/{application_number}/docfiles.txt"

    try:
        response = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)
        document_text = response["Body"].read().decode("utf-8")
    except Exception as e:
        print(f"⚠️ Failed to fetch {s3_key}: {e}")
        return

    vectors = []
    for idx, chunk in enumerate(chunk_text(document_text)):
        embedding = embed_text(chunk)
        chunk_id = f"{application_number}_chunk_{idx}"
        vectors.append({
            "id": chunk_id,
            "values": embedding
        })

    # Reduced batch size to keep each request under Pinecone's 4MB limit.
    BATCH_SIZE = 10
    for i in range(0, len(vectors), BATCH_SIZE):
        batch = vectors[i:i + BATCH_SIZE]
        index.upsert(vectors=batch)

    print(f"✅ Indexed {len(vectors)} chunks for document {application_number}")

def index_all_projects():
    response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix="sample_projects_20250110/", Delimiter='/')

    if "CommonPrefixes" not in response:
        print("⚠️ No projects found.")
        return

    for prefix in response["CommonPrefixes"]:
        application_number = prefix['Prefix'].strip('/').split('/')[-1]
        process_and_index_document(application_number)

if __name__ == "__main__":
    index_all_projects()
