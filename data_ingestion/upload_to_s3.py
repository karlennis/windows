import boto3
import os
from utils.config import S3_BUCKET, S3_REGION

# Initialize S3 client
s3 = boto3.client('s3', region_name=S3_REGION)

def upload_folder_to_s3(base_folder, application_number):
    """
    Recursively uploads all files inside the given base folder to S3 under a prefix matching application_number.
    This preserves subfolder structure inside S3.
    """
    for root, _, files in os.walk(base_folder):
        for file in files:
            local_path = os.path.join(root, file)

            # Compute S3 key (relative path under application_number)
            relative_path = os.path.relpath(local_path, base_folder)
            s3_key = f"{application_number}/{relative_path.replace('\\', '/')}"  # Handle Windows backslashes for S3

            s3.upload_file(local_path, S3_BUCKET, s3_key)
            print(f"Uploaded {local_path} to s3://{S3_BUCKET}/{s3_key}")

if __name__ == "__main__":
    # Locate the `sample_projects_20250110` folder in `data`
    script_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    base_folder = os.path.join(project_root, 'data', 'sample_projects_20250110')

    application_number = "sample_projects_20250110"

    if os.path.exists(base_folder):
        upload_folder_to_s3(base_folder, application_number)
    else:
        print(f"Error: Folder {base_folder} does not exist.")
