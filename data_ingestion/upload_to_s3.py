import boto3
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import S3_BUCKET, S3_REGION
from dotenv import load_dotenv
load_dotenv()


s3 = boto3.client('s3', region_name=S3_REGION)

def upload_docfiles_to_s3(renamed_folder_path, application_number, s3_prefix):
    target_file = os.path.join(renamed_folder_path, "docfiles.txt")
    if os.path.isfile(target_file):
        s3_key = f"{s3_prefix}/{application_number}/docfiles.txt"
        s3.upload_file(target_file, S3_BUCKET, s3_key)
        print(f"☁️ Uploaded to s3://{S3_BUCKET}/{s3_key}")
    else:
        print(f"⚠️ No docfiles.txt found in {renamed_folder_path}")

def upload_all(s3_prefix):
    script_dir = os.path.dirname(__file__)
    docfiles_dir = os.path.join(script_dir, "renameScript", "docfiles")

    for folder_name in os.listdir(docfiles_dir):
        folder_path = os.path.join(docfiles_dir, folder_name)
        if os.path.isdir(folder_path):
            upload_docfiles_to_s3(folder_path, folder_name, s3_prefix)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Error: Missing S3 prefix argument.")
        sys.exit(1)

    s3_prefix = sys.argv[1]
    upload_all(s3_prefix)
