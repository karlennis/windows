import os
import sys
import pandas as pd
import paramiko
import boto3
from dotenv import load_dotenv

# === Load Environment ===
load_dotenv()
S3_BUCKET = os.getenv("S3_BUCKET")
S3_REGION = os.getenv("S3_REGION")

# === SFTP Config ===
HOST = os.getenv("SFTP_HOST")
USERNAME = os.getenv("SFTP_USERNAME")
KEY_FILE = os.path.normpath(os.getenv("SFTP_KEY_FILE"))
KEY_PASSPHRASE = None

# === Paths ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(SCRIPT_DIR, "self-build-2025.csv")
TEMP_DIR = os.path.join(SCRIPT_DIR, "temp_dl")

# === Init AWS S3 ===
s3 = boto3.client("s3", region_name=S3_REGION)

def upload_to_s3(local_file, planning_id, s3_prefix):
    s3_key = f"{s3_prefix}/{planning_id}/docfiles.txt"
    s3.upload_file(local_file, S3_BUCKET, s3_key)
    print(f"☁️ Uploaded to s3://{S3_BUCKET}/{s3_key}")

def main(s3_prefix):
    df = pd.read_csv(CSV_FILE)
    total_docs = len(df)
    df = df.dropna(subset=['filesdir']).copy()
    non_null_dirs = len(df)

    status_list = []
    empty_flags = []
    not_found_count = 0
    uploaded_count = 0
    empty_count = 0

    os.makedirs(TEMP_DIR, exist_ok=True)

    # Connect to SFTP
    key = paramiko.RSAKey.from_private_key_file(KEY_FILE, password=KEY_PASSPHRASE)
    transport = paramiko.Transport((HOST, 22))
    transport.connect(username=USERNAME, pkey=key)
    sftp = paramiko.SFTPClient.from_transport(transport)

    for _, row in df.iterrows():
        planning_id = str(row["planning_id"])
        base_remote = row["filesdir"].strip().replace("\\", "/")
        remote_txt = f"{base_remote}/docfiles.txt"

        local_folder = os.path.join(SCRIPT_DIR, "docfiles", planning_id)
        os.makedirs(local_folder, exist_ok=True)
        local_file = os.path.join(local_folder, "docfiles.txt")

        try:
            # 1) Try the single file first
            sftp.get(remote_txt, local_file)
            with open(local_file, 'r', encoding='utf-8', errors='ignore') as f:  # 🔧
                contents = f.read().strip()  # 🔧
                is_empty = (contents == "")
            if is_empty:
                empty_flags.append(True)
                empty_count += 1
            else:
                empty_flags.append(False)

            status_list.append("found")
            upload_to_s3(local_file, planning_id, s3_prefix)
            uploaded_count += 1

        except FileNotFoundError:
            print(f"⚠️ {remote_txt} not found — attempting to consolidate from {base_remote}/docfiles/")
            remote_folder = f"{base_remote}/docfiles"
            try:
                filenames = sftp.listdir(remote_folder)
                if not filenames:
                    raise FileNotFoundError("Empty folder")
                all_texts = []
                for fname in filenames:
                    remotepath = f"{remote_folder}/{fname}"
                    try:
                        with sftp.open(remotepath, 'r') as rf:
                            data = rf.read()
                            if isinstance(data, bytes):
                                data = data.decode('utf-8', errors='ignore')
                            all_texts.append(data)
                    except Exception as e:
                        print(f"⚠️ Could not read {remotepath}: {e}")

                if all_texts:
                    combined_text = "\n".join(all_texts).strip()  # 🔧
                    is_empty = (combined_text == "")              # 🔧
                    if is_empty:
                        empty_flags.append(True)
                        empty_count += 1
                    else:
                        empty_flags.append(False)

                    with open(local_file, 'w', encoding='utf-8') as lf:
                        lf.write(combined_text)
                        lf.write("\n")
                    status_list.append("generated")
                    upload_to_s3(local_file, planning_id, s3_prefix)
                    uploaded_count += 1
                else:
                    print(f"⚠️ No readable files in {remote_folder}")
                    status_list.append("not found")
                    empty_flags.append(None)  # 🔧 No file to check
                    not_found_count += 1

            except (IOError, FileNotFoundError) as e:
                print(f"⚠️ Folder not found or empty: {remote_folder}")
                status_list.append("not found")
                empty_flags.append(None)  # 🔧
                not_found_count += 1

        except Exception as e:
            print(f"❌ Failed for {planning_id}: {e}")
            status_list.append("not found")
            empty_flags.append(None)  # 🔧
            not_found_count += 1

    sftp.close()
    transport.close()

    # Add status and empty flag to DataFrame
    df["docfiles_status"] = status_list
    df["docfiles_empty"] = empty_flags  # 🔧
    updated_csv = CSV_FILE.replace(".csv", "_with_status.csv")
    df.to_csv(updated_csv, index=False)
    print(f"\n📝 Updated CSV written to: {updated_csv}")

    # === Summary ===
    print("\n📊 Summary:")
    print(f"  Total application spec entries         : {total_docs}")
    print(f"  With non-null 'filesdir' entries       : {non_null_dirs}")
    print(f"  ✅ Successfully found/generated         : {uploaded_count}")
    print(f"  ❌ Not found or failed to download      : {not_found_count}")
    print(f"  ⚠️ Empty 'docfiles.txt' files           : {empty_count}")  # 🔧
    print("✅ Processing complete.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Error: Missing S3 prefix argument.")
        sys.exit(1)
    main(sys.argv[1])
