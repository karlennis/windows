import os
from dotenv import load_dotenv

load_dotenv()
S3_BUCKET = os.getenv("S3_BUCKET")
S3_REGION = os.getenv("S3_REGION")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
