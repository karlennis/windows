import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from .query_service_pipeline import query_pipeline

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Single `/query` route for all requests
@app.route('/query', methods=['POST'])
def handle_query():
    data = request.json  # ✅ Must be a valid JSON object

    if not data:
        return jsonify({"error": "Missing JSON body"}), 400

    user_query = data.get("search_query")
    api_params = data.get("api_params", {})
    is_report = data.get("report", False)

    if not user_query:
        return jsonify({"error": "Query is required"}), 400

    try:
        response = query_pipeline({
            "search_query": user_query,
            "api_params": api_params,
            "report": is_report
        })
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Run the API on the correct port
if __name__ == '__main__':
    app.run()
