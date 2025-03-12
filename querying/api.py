import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from .query_service import generate_answer, generate_report, search_pinecone  # Import functions

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow frontend to interact with backend

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Flask API is running! Use /query or /report."})

# Route to handle AI queries
@app.route('/query', methods=['POST'])
def handle_query():
    """Handles a user query and returns an AI-generated response."""
    data = request.json
    user_query = data.get("query")

    if not user_query:
        return jsonify({"error": "Query is required"}), 400

    try:
        # Search for relevant data
        project_ids, chunk_data = search_pinecone(user_query)
        if not project_ids:
            return jsonify({"error": "No relevant projects found"}), 404

        # Generate AI response
        ai_response = generate_answer(user_query, chunk_data)

        return jsonify({"query": user_query, "response": ai_response})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Route to handle AI reports
@app.route('/report', methods=['POST'])
def generate_ai_report():
    """Generates a detailed report based on a user query."""
    data = request.json
    user_query = data.get("query")

    if not user_query:
        return jsonify({"error": "Query is required"}), 400

    try:
        # Search for relevant data
        project_ids, chunk_data = search_pinecone(user_query)
        if not project_ids:
            return jsonify({"error": "No relevant projects found"}), 404

        # Generate AI report
        ai_report = generate_report(user_query, project_ids, chunk_data)

        return jsonify({"query": user_query, "report": ai_report})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the API on the correct port
if __name__ == '__main__':
    app.run()
