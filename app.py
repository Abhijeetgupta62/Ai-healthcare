"""
Healthcare Chatbot — Flask Backend
Serves the chat UI and the /chat API endpoint
"""

from flask import Flask, render_template, request, jsonify
from response_generator import HealthcareResponder
import os

app = Flask(__name__)
bot = HealthcareResponder()   # loads RAG index once at startup


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data    = request.get_json(force=True)
    message = data.get("message", "").strip()
    if not message:
        return jsonify({"response": "Please enter a message.", "status": "error"}), 400
    try:
        response = bot.respond(message)
        return jsonify({"response": response, "status": "ok"})
    except Exception as e:
        return jsonify({"response": f"Error: {str(e)}", "status": "error"}), 500


@app.route("/diseases")
def diseases():
    import json
    with open("healthcare_dataset.json", encoding="utf-8") as f:
        data = json.load(f)
    return jsonify([{"name": d["disease"], "category": d["category"]} for d in data])


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n🏥 Healthcare Chatbot running on http://localhost:{port}\n")
    app.run(debug=False, host="0.0.0.0", port=port)
