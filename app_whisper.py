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
    default_limits=["60 per minute"],
    storage_uri=os.getenv("REDIS_URL", "memory://"),
)




load_dotenv()

app = Flask(__name__)
limiter = Limiter(app, key_func=get_remote_address, default_limits=["60 per minute"])

OPENAI_API_KEYs = os.getenv("OPENAI_API_KEYs")
PROXY_SECRET = os.getenv("PROXY_SECRET")


@app.route("/transcribe", methods=["POST"])
@limiter.limit("30 per minute")
def transcribe():
    """
    Transcribe audio using OpenAI Whisper API
    Expects JSON with base64-encoded audio and model name
    
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
        key = os.getenv("OPENAI_API_KEYs")
        if not key:
            return jsonify({"error": "openai-key-not-configured"}), 500

        # Require proxy secret if set
        if PROXY_SECRET:
            header = request.headers.get("X-EXT-SECRET") or request.headers.get("x-ext-secret")
            if not header or header != PROXY_SECRET:
                return jsonify({"error": "unauthorized"}), 401

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
                    "Authorization": f"Bearer {key}",
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
