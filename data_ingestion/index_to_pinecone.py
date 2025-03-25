import boto3
from pinecone import Pinecone
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import S3_BUCKET, S3_REGION

# Load environment variables
load_dotenv()

# === CONFIGURATION ===
INDEX_NAME = "planning-docs"
EMBEDDING_DIM = 384  # For 'all-MiniLM-L6-v2'

# === Initialize Clients ===
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
s3 = boto3.client("s3", region_name=S3_REGION)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# === Ensure the index exists ===
existing_indexes = [idx.name for idx in pc.list_indexes()]
if INDEX_NAME not in existing_indexes:
    print(f"❌ Index '{INDEX_NAME}' does not exist. Please create it manually in the Pinecone Console.")
    sys.exit(1)

# === Confirm wipe ===
print(f"\n⚠️  You are about to CLEAR all data from the Pinecone index: '{INDEX_NAME}'")
confirm = input("Type 'yes' to continue or anything else to cancel: ").strip().lower()
if confirm != "yes":
    print("❌ Cancelled. No data was deleted.")
    sys.exit(0)

print(f"🧽 Clearing index '{INDEX_NAME}'...")
pc.Index(INDEX_NAME).delete(delete_all=True)
index = pc.Index(INDEX_NAME)

# === Embedding + Indexing Logic ===
def embed_text(text):
    return embedding_model.encode(text, normalize_embeddings=True).tolist()

def chunk_text(text, max_words=200):
    words = text.split()
    for i in range(0, len(words), max_words):
        yield " ".join(words[i:i + max_words])

def process_and_index_document(application_number, s3_prefix):
    s3_key = f"{s3_prefix}/{application_number}/docfiles.txt"
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

    BATCH_SIZE = 10
    for i in range(0, len(vectors), BATCH_SIZE):
        index.upsert(vectors=vectors[i:i + BATCH_SIZE])

    print(f"✅ Indexed {len(vectors)} chunks for {application_number}")

def index_all_projects(s3_prefix):
    response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=f"{s3_prefix}/", Delimiter="/")

    if "CommonPrefixes" not in response:
        print(f"⚠️ No folders found in S3 at prefix: {s3_prefix}")
        return

    for prefix in response["CommonPrefixes"]:
        application_number = prefix['Prefix'].strip('/').split('/')[-1]
        process_and_index_document(application_number, s3_prefix)

# === Entry Point ===
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Please provide the S3 prefix as an argument.")
        sys.exit(1)

    s3_prefix = sys.argv[1]
    index_all_projects(s3_prefix)
