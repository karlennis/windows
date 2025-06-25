import os
import json
import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from .query_service_pipeline import query_pipeline

# ──────────────── Environment & Flask app ─────────────────────────────────────
load_dotenv()

app = Flask(__name__)
CORS(app)                      # Enable CORS for the Angular front-end

# ──────────────── Helper: write backend result to file ───────────────────────
def _dump_response_to_file(data: dict) -> None:
    """
    Write the exact JSON object returned by query_pipeline
    to a timestamped text file in the same directory as api.py.
    """
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(os.path.dirname(__file__),
                        f"last_backend_output_{ts}.txt")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# ──────────────── /query route ───────────────────────────────────────────────
@app.route("/query", methods=["POST"])
def handle_query():
    data = request.get_json(silent=True)          # safer than request.json
    if not data:
        return jsonify({"error": "Missing JSON body"}), 400

    user_query = data.get("search_query", "").strip()
    if not user_query:
        return jsonify({"error": "Query is required"}), 400

    try:
        response = query_pipeline(
            {
                "search_query": user_query,
                "api_params" : data.get("api_params", {}),
                "report"     : data.get("report", False),
            }
        )


        return jsonify(response)

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

# ──────────────── Run locally ────────────────────────────────────────────────
if __name__ == "__main__":
    app.run()
