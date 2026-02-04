from flask import Flask, request, jsonify
from flask_cors import CORS

from pipeline import run_pipeline

app = Flask(__name__)
CORS(app)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/api/chat")
def chat():
    data = request.get_json(force=True) or {}
    user_message = (data.get("message") or "").strip()

    if not user_message:
        return jsonify({"error": "message is required"}), 400

    result = run_pipeline(user_message)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
