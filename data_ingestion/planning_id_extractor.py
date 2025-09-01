import os
import requests
from dotenv import load_dotenv

# ─── Optional: load variables from a local “.env” file ───────────────────────
# (make sure you pip-install python-dotenv if you use this)
load_dotenv()

# ─── API details from environment ─────────────────────────────────────────────
BASE_URL = "https://api12.buildinginfo.com/api/v2/bi/projects/t-projects"
API_KEY = os.getenv("BUILDING_INFO_API_KEY")
UKEY   = os.getenv("BUILDING_INFO_API_UKEY")

if not API_KEY or not UKEY:
    raise EnvironmentError("Please set BUILDING_INFO_API_KEY and BUILDING_INFO_API_UKEY in your environment")

# ─── Query parameters ─────────────────────────────────────────────────────────
PARAMS = (
    f"api_key={API_KEY}"
    f"&ukey={UKEY}"
    # f"&category=11"
    f"&_apion=1"
    # f"&min_apion=2024-01-01"
    # f"&max_apion=2024-12-31"
)

def fetch_all_planning_ids():
    offset = 0
    limit = 1000
    all_ids = []

    while True:
        paged_url = f"{BASE_URL}?{PARAMS}&more=limit {offset},{limit}&order=planning_id"
        print(f"🔄 Fetching: offset {offset}")

        resp = requests.get(paged_url)
        if resp.status_code != 200:
            print(f"❌ Error: {resp.status_code}", resp.text)
            break

        rows = resp.json().get("data", {}).get("rows", [])
        if not rows:
            break

        ids_batch = [str(r["planning_id"]) for r in rows if r.get("planning_id")]
        all_ids.extend(ids_batch)
        offset += limit

    return all_ids

if __name__ == "__main__":
    planning_ids = fetch_all_planning_ids()

    output_file = "planning_ids.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        if planning_ids:
            f.write("✅ Paste this in your SQL IN clause:\n")
            f.write(", ".join(planning_ids))
            print(f"\n✅ Wrote {len(planning_ids)} planning IDs to '{output_file}'.")
        else:
            msg = "⚠️ No planning IDs found."
            f.write(msg)
            print(msg)
