import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
import requests
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

load_dotenv()

app = Flask(__name__)

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["60 per minute"]
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PROXY_SECRET = os.getenv("PROXY_SECRET")


@app.route("/", methods=["GET"])
def index():
    return "Dux AutoReply Flask proxy running"


@app.route("/generate", methods=["POST"])
@limiter.limit("60 per minute")
def generate():
    try:
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            return jsonify({"error": "openai-key-not-configured"}), 500

        if PROXY_SECRET:
            header = request.headers.get("X-EXT-SECRET") or request.headers.get("x-ext-secret")
            if not header or header != PROXY_SECRET:
                return jsonify({"error": "unauthorized"}), 401

        data = request.get_json() or {}
        email_content = data.get("emailContent", "")
        model = data.get("model", "gpt-4o-mini")
        messages = data.get("messages") or [
            {"role": "user", "content": f"Write a concise reply for this email:\n\n{email_content}"}
        ]

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": data.get("max_tokens", 400),
        }

        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        result = resp.json()

        reply = None
        choices = result.get("choices")
        if choices:
            first = choices[0]
            message = first.get("message") or first.get("text")
            if isinstance(message, dict):
                reply = message.get("content")
            elif isinstance(message, str):
                reply = message

        return jsonify({"raw": result, "reply": reply})

    except requests.RequestException as e:
        app.logger.exception("proxy error")
        return jsonify({"error": "proxy-error", "details": str(e)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
