import os
import sys
import openai
import boto3
from dotenv import load_dotenv
from nltk.tokenize import sent_tokenize
from pinecone import Pinecone
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import S3_BUCKET, S3_REGION

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize Pinecone + S3
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "openai-docs"
index = pc.Index(index_name)
s3 = boto3.client("s3", region_name=S3_REGION)

# Constants: target ~300-word chunks with 50-word overlap
NAMESPACE = "default"
MAX_WORDS_PER_CHUNK = 300
OVERLAP_WORDS = 50

# Function to chunk text into ~300-word segments with overlap,
# respecting sentence boundaries.
def chunk_text(text, max_words=MAX_WORDS_PER_CHUNK, overlap_words=OVERLAP_WORDS):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk_words = []
    current_word_count = 0

    for sentence in sentences:
        # Split sentence into words
        words = sentence.split()
        # If a single sentence is ridiculously long, split it by words.
        if len(words) > max_words:
            # Flush the current chunk if not empty
            if current_chunk_words:
                chunks.append(" ".join(current_chunk_words))
                current_chunk_words = []
                current_word_count = 0
            for i in range(0, len(words), max_words):
                sub_chunk = words[i:i + max_words]
                chunks.append(" ".join(sub_chunk))
            continue

        # If adding this sentence would exceed our chunk size, finalize the chunk.
        if current_word_count + len(words) > max_words:
            chunks.append(" ".join(current_chunk_words))
            # Build overlap: last overlap_words from the current chunk
            overlap = current_chunk_words[-overlap_words:] if len(current_chunk_words) > overlap_words else current_chunk_words
            current_chunk_words = overlap.copy()
            current_word_count = len(current_chunk_words)

        current_chunk_words.extend(words)
        current_word_count += len(words)

    if current_chunk_words:
        chunks.append(" ".join(current_chunk_words))
    return chunks

# Embedding helper: embeds chunks in batches.
def embed_text(texts, batch_size=10):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            response = openai.embeddings.create(
                input=batch,
                model="text-embedding-3-small"
            )
            all_embeddings.extend([d.embedding for d in response.data])
        except Exception as e:
            print(f"❌ Embedding failed for batch starting at {i}: {e}")
    return all_embeddings

# Helper to upsert vectors in smaller batches
def upsert_vectors_in_batches(index, vectors, namespace, batch_size=100):
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        try:
            index.upsert(vectors=batch, namespace=namespace)
            print(f"✅ Upserted batch {i // batch_size + 1} with {len(batch)} vectors")
        except Exception as e:
            print(f"❌ Upsert failed for batch starting at {i}: {e}")

# Main indexing logic
def process_and_index_document(project_id, s3_prefix):
    s3_key = f"{s3_prefix}/{project_id}/docfiles.txt"
    try:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)
        text = obj["Body"].read().decode("utf-8")
    except Exception as e:
        print(f"❌ Failed to fetch {s3_key}: {e}")
        return

    chunks = chunk_text(text)
    print(f"✅ Created {len(chunks)} chunks for project {project_id}")
    embeddings = embed_text(chunks)

    to_upsert = []
    for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):
        to_upsert.append({
            "id": f"{project_id}_chunk_{i}",
            "values": vector,
            "metadata": {
                "project_id": project_id,
            }
        })

    upsert_vectors_in_batches(index, to_upsert, NAMESPACE)
    print(f"✅ Indexed {len(to_upsert)} chunks for project {project_id}")

def index_all_projects(s3_prefix):
    response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=f"{s3_prefix}/", Delimiter='/')
    if "CommonPrefixes" not in response:
        print("⚠️ No subfolders found.")
        return

    for prefix in response["CommonPrefixes"]:
        project_id = prefix['Prefix'].strip('/').split('/')[-1]
        process_and_index_document(project_id, s3_prefix)

# CLI Entry Point
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Provide the S3 prefix (e.g., planning_documents_2025_04)")
        sys.exit(1)

    s3_prefix = sys.argv[1]

    print(f"⚠️ You are about to CLEAR all data from the Pinecone index: '{index_name}' (namespace: '{NAMESPACE}')")
    confirm = input("Type 'yes' to continue or anything else to cancel: ").strip().lower()
    if confirm == "yes":
        try:
            index.delete(delete_all=True, namespace=NAMESPACE)
            print(f"🧽 Index '{index_name}' cleared in namespace '{NAMESPACE}'")
        except Exception as e:
            print(f"❌ Failed to clear index: {e}")
            sys.exit(1)
    else:
        print("❌ Operation cancelled by user.")
        sys.exit(0)

    print(f"🚀 Indexing documents from: {s3_prefix}")
    index_all_projects(s3_prefix)
