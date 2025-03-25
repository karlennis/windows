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
    """Handles AI queries and reports."""
    data = request.json
    user_query = data.get("query")

    if not user_query:
        return jsonify({"error": "Query is required"}), 400

    try:
        response = query_pipeline(user_query)
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the API on the correct port
if __name__ == '__main__':
    app.run()
