# Windows Planning Application Query System

## 📌 Project Overview
This project enables querying of **Windows Planning Application Documents** stored in an **S3 bucket** using a **Retrieval-Augmented Generation (RAG)** approach. It allows users to:

✅ Upload planning documents to **AWS S3**
✅ Index documents into **Pinecone** for fast vector search
✅ Use **local embeddings (MiniLM)** to reduce costs
✅ Retrieve relevant documents and generate **AI-powered answers** via **OpenAI GPT-4**

---

## 🏗 **System Architecture**

1️⃣ **Data Ingestion & Indexing:**
   - Planning documents (e.g., `docfiles.txt`) are stored in **AWS S3 (Org Bucket)**.
   - The system downloads, **chunks**, and embeds documents.
   - The generated vectors are **stored in Pinecone** for fast retrieval.

2️⃣ **Query Processing & Retrieval:**
   - Users submit **natural language queries** via the CLI.
   - Queries are embedded using **MiniLM** and searched in **Pinecone**.
   - Relevant document chunks are fetched from **S3** and passed to **OpenAI GPT-4**.
   - The AI generates an **answer based on retrieved documents**.

---

## 📦 **System Requirements**

### **1️⃣ Install Dependencies**
Ensure you have Python **3.9+** installed, then install all required dependencies:
```bash
pip install -r requirements.txt
```

### **2️⃣ Configure AWS Credentials**
If using an **organization's S3 bucket**, set up a **named profile**:
```bash
aws configure --profile org-profile
```
Or update your `~/.aws/credentials` file manually:
```ini
[org-profile]
aws_access_key_id=YOUR_ACCESS_KEY
aws_secret_access_key=YOUR_SECRET_KEY
region=YOUR_REGION
```

### **3️⃣ Set Environment Variables**
Create a `.env` file with the following:
```ini
S3_BUCKET=your-org-bucket-name
S3_REGION=your-region
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_ENV=your-pinecone-env
OPENAI_API_KEY=your-openai-api-key
```

---

## 🚀 **Running the System**

### **1️⃣ Upload Planning Documents to S3**
If documents are not yet uploaded, use the S3 CLI:
```bash
aws s3 cp data/sample_projects_20250110 s3://your-org-bucket-name/sample_projects_20250110 --recursive
```

### **2️⃣ Index Documents into Pinecone**
```bash
python -m data_ingestion.index_to_pinecone
```
✅ This will:
- Download `docfiles.txt` from S3.
- Split documents into **200-word chunks**.
- Generate embeddings and **store them in Pinecone**.
- **Batch uploads to avoid Pinecone’s 4MB limit.**

### **3️⃣ Query the Documents**
```bash
python -m querying.query_service
```
✅ This will:
- Accept **natural language queries**.
- Retrieve **relevant document chunks** from Pinecone.
- Fetch **full text from S3 if needed**.
- Generate **AI-powered responses using GPT-4**.

Example:
```
🔎 Ask a question: What projects use aluminum windows?

🔍 **Top Relevant Chunks:**
- 1500004_chunk_87
- 1500002_chunk_50
- 1500003_chunk_51

📥 Fetched: sample_projects_20250110/1500004/docfiles.txt
📥 Fetched: sample_projects_20250110/1500002/docfiles.txt

🧠 **AI Response:**
Projects 1500004 and 1500002 use aluminum windows due to energy efficiency.
```


### 📜 **Summary**
✅ **AWS S3** stores planning documents.
✅ **Sentence-Transformers (MiniLM)** generates free, fast embeddings.
✅ **Pinecone** enables vector search over document chunks.
✅ **GPT-4** generates answers based on retrieved documents.
✅ CLI-based querying enables interactive search.

