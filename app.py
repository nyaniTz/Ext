import os
import base64
import io
import time
from dotenv import load_dotenv
from flask import Flask, request, jsonify
import requests
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

load_dotenv()

app = Flask(__name__)

# Configure rate limiter with Redis if available, otherwise memory
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["60 per minute"],
    storage_uri=os.getenv("REDIS_URL", "memory://"),

)


OPENAI_API_KEY = os.getenv("OPENAI_API_KEYs")  # Note: keeping your var name
PROXY_SECRET = os.getenv("PROXY_SECRET")


def check_auth():
    """Check proxy secret authorization if configured"""
    if PROXY_SECRET:
        header = request.headers.get("X-EXT-SECRET") or request.headers.get("x-ext-secret")
        if not header or header != PROXY_SECRET:
            return jsonify({"error": "unauthorized"}), 401
    return None


@app.route("/", methods=["GET"])
def index():
    return "NEU AutoReply Flask proxy running"


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"}), 200


@app.route("/generate", methods=["POST"])
@limiter.limit("60 per minute")
def generate():
    """Generate email reply using OpenAI Chat API"""
    try:
        if not OPENAI_API_KEY:
            return jsonify({"error": "openai-key-not-configured"}), 500

        auth_error = check_auth()
        if auth_error:
            return auth_error

        data = request.get_json() or {}
        email_content = data.get("emailContent", "")
        model = data.get("model") or os.getenv("DEFAULT_MODEL", "gpt-3.5-turbo")
        messages = data.get("messages") or [
            {"role": "user", "content": f"Write a concise reply for this email:\n\n{email_content}"}
        ]

        # Build payload
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": data.get("max_tokens", 400)
        }
        
        # Support multiple completions
        if data.get("n"):
            payload["n"] = int(data.get("n"))
        elif data.get("multi"):
            payload["n"] = 2

        # Retry loop for rate limits
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
                app.logger.exception("Request to OpenAI failed on attempt %s", attempt + 1)
                return jsonify({"error": "proxy-error", "details": str(e)}), 502

            # Handle rate limiting
            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                try:
                    wait = float(retry_after) if retry_after else (backoff_base * (2 ** attempt))
                except ValueError:
                    wait = backoff_base * (2 ** attempt)
                
                app.logger.warning("OpenAI returned 429; attempt=%s waiting=%s", attempt + 1, wait)
                time.sleep(wait)
                continue

            break

        if resp is None:
            return jsonify({"error": "proxy-error", "details": "no-response-from-openai"}), 502

        # Parse response
        try:
            result = resp.json()
        except ValueError:
            result = {"error": "non-json-response", "status_text": resp.text}

        if resp.status_code >= 400:
            app.logger.error("OpenAI error: %s", result)
            return jsonify({"error": "openai_error", "details": result}), resp.status_code

        # Extract replies
        reply = None
        replies = []
        choices = result.get("choices", [])
        
        if isinstance(choices, list) and len(choices) > 0:
            for ch in choices:
                message = ch.get("message") or ch.get("text")
                content = None
                
                if isinstance(message, dict):
                    content = message.get("content")
                elif isinstance(message, str):
                    content = message
                    
                if content:
                    replies.append(content)
                    
            if replies:
                reply = replies[0]

        return jsonify({"raw": result, "reply": reply, "replies": replies})
        
    except Exception as e:
        app.logger.exception("proxy error in /generate")
        return jsonify({"error": "proxy-error", "details": str(e)}), 500


@app.route("/transcribe", methods=["POST"])
@limiter.limit("30 per minute")
def transcribe():
    """Transcribe audio using OpenAI Whisper API
    
    Request format:
    {
        "audio": "base64-encoded-audio-data",
        "model": "whisper-1"  (optional, defaults to whisper-1)
    }
    
    Response format:
    {
        "text": "transcribed text",
        "raw": {OpenAI API response}
    }
    """
    try:
        if not OPENAI_API_KEY:
            return jsonify({"error": "openai-key-not-configured"}), 500

        auth_error = check_auth()
        if auth_error:
            return auth_error

        data = request.get_json() or {}
        audio_base64 = data.get("audio", "")
        model = data.get("model", "whisper-1")

        if not audio_base64:
            return jsonify({"error": "no-audio-provided"}), 400

        # Decode base64 audio
        try:
            audio_bytes = base64.b64decode(audio_base64)
        except Exception as e:
            app.logger.error("Failed to decode base64 audio: %s", str(e))
            return jsonify({"error": "invalid-audio-format"}), 400

        # Create file-like object for multipart upload
        audio_file = io.BytesIO(audio_bytes)
        
        # Send to OpenAI Whisper API
        files = {"file": ("audio.webm", audio_file, "audio/webm")}
        data_payload = {"model": model}

        try:
            resp = requests.post(
                "https://api.openai.com/v1/audio/transcriptions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                },
                files=files,
                data=data_payload,
                timeout=30,
            )
        except requests.RequestException as e:
            app.logger.exception("Request to OpenAI Whisper failed")
            return jsonify({"error": "proxy-error", "details": str(e)}), 502

        # Parse response
        try:
            result = resp.json()
        except ValueError:
            result = {"error": "non-json-response", "status_text": resp.text}

        if resp.status_code >= 400:
            app.logger.error("OpenAI Whisper error: %s", result)
            return jsonify({"error": "openai_error", "details": result}), resp.status_code

        # Extract transcription text
        text = result.get("text", "")
        
        return jsonify({"text": text, "raw": result})
        
    except Exception as e:
        app.logger.exception("transcribe error")
        return jsonify({"error": "proxy-error", "details": str(e)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
