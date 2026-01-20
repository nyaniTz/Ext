import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
import requests
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

load_dotenv()

app = Flask(__name__)
limiter = Limiter(app, key_func=get_remote_address, default_limits=["60 per minute"])

OPENAI_API_KEYs = os.getenv("OPENAI_API_KEYs")
PROXY_SECRET = os.getenv("PROXY_SECRET")


@app.route("/", methods=["GET"])
def index():
    return "NEU AutoReply Flask proxy running"


@app.route("/generate", methods=["POST"])
@limiter.limit("60 per minute")
def generate():
    try:
        key = os.getenv("OPENAI_API_KEYs")
        if not key:
            return jsonify({"error": "openai-key-not-configured"}), 500

        # Require proxy secret if set
        if PROXY_SECRET:
            header = request.headers.get("X-EXT-SECRET") or request.headers.get("x-ext-secret")
            if not header or header != PROXY_SECRET:
                return jsonify({"error": "unauthorized"}), 401

        data = request.get_json() or {}
        email_content = data.get("emailContent", "")
        # Allow overriding model per-request; default to cheaper model via env to save quota
        model = data.get("model") or os.getenv("DEFAULT_MODEL", "gpt-3.5-turbo")
        messages = data.get("messages") or [
            {"role": "user", "content": f"Write a concise reply for this email:\n\n{email_content}"}
        ]

        # Support requesting multiple completions (n) from OpenAI if caller provided 'n' or 'multi'
        payload = {"model": model, "messages": messages, "max_tokens": data.get("max_tokens", 400)}
        if data.get("n"):
            payload["n"] = int(data.get("n"))
        elif data.get("multi"):
            payload["n"] = 2

        # Retry loop for transient 429 responses
        max_retries = 3
        backoff_base = 1.0
        resp = None
        for attempt in range(max_retries):
            try:
                resp = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=30,
                )
            except requests.RequestException as e:
                app.logger.exception("Request to OpenAI failed on attempt %s", attempt + 1)
                # On network errors, break and return proxy error
                return jsonify({"error": "proxy-error", "details": str(e)}), 502

            # If rate limited, honor Retry-After header if present, otherwise exponential backoff
            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                try:
                    wait = float(retry_after) if retry_after is not None else (backoff_base * (2 ** attempt))
                except ValueError:
                    wait = backoff_base * (2 ** attempt)
                app.logger.warning("OpenAI returned 429; attempt=%s waiting=%s", attempt + 1, wait)
                # Sleep before next attempt
                import time

                time.sleep(wait)
                continue

            # not a 429, break the retry loop
            break

        if resp is None:
            return jsonify({"error": "proxy-error", "details": "no-response-from-openai"}), 502

        # Parse JSON if possible, otherwise capture text
        try:
            result = resp.json()
        except ValueError:
            result = {"error": "non-json-response", "status_text": resp.text}

        if resp.status_code >= 400:
            # Forward OpenAI error payload and use the same status code
            app.logger.error("OpenAI error: %s", result)
            return jsonify({"error": "openai_error", "details": result}), resp.status_code

        # Try to extract assistant reply text(s) if present. Return a 'replies' array when multiple choices requested.
        reply = None
        replies = []
        choices = result.get("choices")
        if choices and isinstance(choices, list) and len(choices) > 0:
            for ch in choices:
                message = ch.get("message") or ch.get("text")
                content = None
                if isinstance(message, dict):
                    content = message.get("content")
                elif isinstance(message, str):
                    content = message
                if content:
                    replies.append(content)
            if len(replies) > 0:
                reply = replies[0]

        return jsonify({"raw": result, "reply": reply, "replies": replies})
    except Exception as e:
        app.logger.exception("proxy error")
        return jsonify({"error": "proxy-error", "details": str(e)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
