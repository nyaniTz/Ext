import os
import time
from dotenv import load_dotenv
from flask import Flask, request, jsonify
import requests
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Load environment variables
load_dotenv()

# Flask app
app = Flask(__name__)

# Flask-Limiter (FIXED for v3+)
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["60 per minute"],
)
limiter.init_app(app)

# Environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEYs")  # keep name as-is to avoid breaking env
PROXY_SECRET = os.getenv("PROXY_SECRET")


@app.route("/", methods=["GET"])
def index():
    return "Dux AutoReply Flask proxy running"


@app.route("/generate", methods=["POST"])
@limiter.limit("60 per minute")
def generate():
    try:
        if not OPENAI_API_KEY:
            return jsonify({"error": "openai-key-not-configured"}), 500

        # Optional proxy secret check
        if PROXY_SECRET:
            header = request.headers.get("X-EXT-SECRET") or request.headers.get("x-ext-secret")
            if header != PROXY_SECRET:
                return jsonify({"error": "unauthorized"}), 401

        data = request.get_json() or {}

        email_content = data.get("emailContent", "")
        model = data.get("model") or os.getenv("DEFAULT_MODEL", "gpt-3.5-turbo")

        messages = data.get("messages") or [
            {
                "role": "user",
                "content": f"Write a concise reply for this email:\n\n{email_content}"
            }
        ]

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": data.get("max_tokens", 400),
        }

        # Retry logic for OpenAI 429
        max_retries = 3
        backoff_base = 1.0
        resp = None

        for attempt in range(max_retries):
            try:
                resp = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {OPENAI_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=30,
                )
            except requests.RequestException as e:
                app.logger.exception("Request to OpenAI failed")
                return jsonify({"error": "proxy-error", "details": str(e)}), 502

            if resp.status_code == 429:
                wait = backoff_base * (2 ** attempt)
                time.sleep(wait)
                continue

            break

        if resp is None:
            return jsonify({"error": "proxy-error", "details": "no-response-from-openai"}), 502

        try:
            result = resp.json()
        except ValueError:
            return jsonify({"error": "non-json-response", "details": resp.text}), 502

        if resp.status_code >= 400:
            return jsonify({"error": "openai_error", "details": result}), resp.status_code

        # Extract reply
        reply = None
        choices = result.get("choices", [])
        if choices:
            msg = choices[0].get("message")
            if msg:
                reply = msg.get("content")

        return jsonify({
            "reply": reply,
            "raw": result
        })

    except Exception as e:
        app.logger.exception("Unhandled proxy error")
        return jsonify({"error": "proxy-error", "details": str(e)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
