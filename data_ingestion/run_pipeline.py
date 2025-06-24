import os
import subprocess
from datetime import datetime

def run_script(script_path, s3_prefix=None):
    print(f"\n🚀 Running: {script_path}")
    cmd = ["python", script_path]
    if s3_prefix:
        cmd.append(s3_prefix)
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    current_prefix = f"planning_documents_{datetime.now().strftime('%Y_%m')}"

    # Step 1: Run scraper (download + upload)
    scraper_script = os.path.join(base_dir, "renameScript", "scraper.py")
    run_script(scraper_script, s3_prefix=current_prefix)

    # Step 2: Index to Pinecone
    index_script = os.path.join(base_dir, "index_to_pinecone.py")
    run_script(index_script, s3_prefix=current_prefix)

    print(f"\n✅ Pipeline complete under S3 prefix: {current_prefix}")
