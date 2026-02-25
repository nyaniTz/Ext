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
# Parse Redis URL to ensure it's valid, fallback to memory if issues
redis_url = os.getenv("REDIS_URL", "").strip()
# Clean up any command prefixes that might have been copied
if redis_url and "redis-cli" in redis_url:
    # Extract just the URL part after any command flags
    parts = redis_url.split()
    for i, part in enumerate(parts):
        if part.startswith("redis://") or part.startswith("rediss://"):
            redis_url = part
            break
    else:
        redis_url = ""

storage_uri = redis_url if redis_url else "memory://"

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["60 per minute"],
    storage_uri=storage_uri,
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEYs")  # Note: keeping your var name
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
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
    """Generate email reply using GPT / DeepSeek / Gemini based on requested model"""
    try:
        auth_error = check_auth()
        if auth_error:
            return auth_error

        data = request.get_json() or {}
        email_content = data.get("emailContent", "")
        raw_model = (data.get("model") or "").strip()
        if raw_model in ("", "auto", None):
            raw_model = os.getenv("DEFAULT_MODEL", "gpt-3.5-turbo")

        model = raw_model
        model_lower = model.lower()

        # Decide provider from model id prefix
        if model_lower.startswith("deepseek"):
            provider = "deepseek"
        elif model_lower.startswith("gemini"):
            provider = "gemini"
        else:
            provider = "openai"

        # Ensure required API key exists for chosen provider
        if provider == "openai" and not OPENAI_API_KEY:
            return jsonify({"error": "openai-key-not-configured"}), 500
        if provider == "deepseek" and not DEEPSEEK_API_KEY:
            return jsonify({"error": "deepseek-key-not-configured"}), 500
        if provider == "gemini" and not GEMINI_API_KEY:
            return jsonify({"error": "gemini-key-not-configured"}), 500

        messages = data.get("messages") or [
            {"role": "user", "content": f"Write a concise reply for this email:\n\n{email_content}"}
        ]

        # Helper: call OpenAI/DeepSeek style chat-completions endpoint
        def call_chat_completions(base_url: str, api_key: str):
            payload = {
                "model": model,
                "messages": messages,
                "max_tokens": data.get("max_tokens", 400),
            }
            # Support multiple completions
            if data.get("n"):
                payload["n"] = int(data.get("n"))
            elif data.get("multi"):
                payload["n"] = 2

            max_retries = 3
            backoff_base = 1.0
            resp_local = None

            for attempt in range(max_retries):
                try:
                    resp_local = requests.post(
                        f"{base_url.rstrip('/')}/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json",
                        },
                        json=payload,
                        timeout=30,
                    )
                except requests.RequestException as e:
                    app.logger.exception("Request to %s failed on attempt %s", base_url, attempt + 1)
                    return None, {"error": "proxy-error", "details": str(e)}

                if resp_local.status_code == 429:
                    retry_after = resp_local.headers.get("Retry-After")
                    try:
                        wait = float(retry_after) if retry_after else (backoff_base * (2 ** attempt))
                    except ValueError:
                        wait = backoff_base * (2 ** attempt)
                    app.logger.warning("%s returned 429; attempt=%s waiting=%s", base_url, attempt + 1, wait)
                    time.sleep(wait)
                    continue
                break

            return resp_local, None

        # Helper: call Gemini generateContent
        def call_gemini_generate():
            # Flatten messages into Gemini format: system prompt + user conversation
            system_parts = []
            user_parts = []
            for m in messages:
                role = (m.get("role") or "").lower()
                content = m.get("content") or ""
                if role == "system":
                    system_parts.append(content)
                else:
                    user_parts.append(f"[{role}] {content}")

            contents = [
                {
                    "parts": [{"text": "\n\n".join(user_parts) or email_content}]
                }
            ]

            body = {"contents": contents}
            if system_parts:
                body["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_parts)}]}

            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
            try:
                resp_local = requests.post(
                    url,
                    headers={"Content-Type": "application/json", "x-goog-api-key": GEMINI_API_KEY},
                    json=body,
                    timeout=30,
                )
            except requests.RequestException as e:
                app.logger.exception("Request to Gemini failed")
                return None, {"error": "proxy-error", "details": str(e)}

            return resp_local, None

        # Route call based on provider
        if provider in ("openai", "deepseek"):
            base_url = "https://api.openai.com" if provider == "openai" else "https://api.deepseek.com"
            api_key = OPENAI_API_KEY if provider == "openai" else DEEPSEEK_API_KEY
            resp, err = call_chat_completions(base_url, api_key)
        else:
            resp, err = call_gemini_generate()

        if err is not None:
            return jsonify(err), 502

        if resp is None:
            return jsonify({"error": "proxy-error", "details": "no-response-from-provider", "provider": provider}), 502

        # Parse response
        try:
            result = resp.json()
        except ValueError:
            result = {"error": "non-json-response", "status_text": resp.text}

        if resp.status_code >= 400:
            app.logger.error("Provider error (%s): %s", provider, result)
            return jsonify({"error": f"{provider}_error", "details": result}), resp.status_code

        # Extract replies – OpenAI / DeepSeek style
        reply = None
        replies = []

        if provider in ("openai", "deepseek"):
            choices = result.get("choices", [])
            if isinstance(choices, list) and len(choices) > 0:
                for ch in choices:
                    message = ch.get("message") or ch.get("text")
                    content = None
                    if isinstance(message, dict):
                        # DeepSeek reasoning model exposes reasoning_content, ignore it for now and use content
                        content = message.get("content")
                    elif isinstance(message, str):
                        content = message
                    if content:
                        replies.append(content)
                if replies:
                    reply = replies[0]
        else:
            # Gemini: candidates[0].content.parts[].text
            candidates = result.get("candidates") or []
            if candidates:
                first = candidates[0] or {}
                content_obj = first.get("content") or {}
                parts = content_obj.get("parts") or []
                texts = []
                for p in parts:
                    t = p.get("text")
                    if t:
                        texts.append(t)
                if texts:
                    reply = "\n".join(texts)
                    replies = [reply]

        return jsonify({"raw": result, "reply": reply, "replies": replies, "provider": provider, "model": model})
        
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


@app.route("/speak", methods=["POST"])
@limiter.limit("30 per minute")
def speak():
    """
    Text-to-speech using OpenAI audio/speech API.

    Request JSON:
    {
      "text": "string to read",
      "model": "gpt-4o-mini-tts",  # optional
      "voice": "alloy"             # optional
    }

    Response JSON:
    {
      "audio": "<base64-encoded audio>",
      "format": "mp3"
    }
    """
    try:
        if not OPENAI_API_KEY:
            return jsonify({"error": "openai-key-not-configured"}), 500

        auth_error = check_auth()
        if auth_error:
            return auth_error

        data = request.get_json() or {}
        text = (data.get("text") or "").strip()
        model = data.get("model") or "gpt-4o-mini-tts"
        voice = data.get("voice") or "alloy"

        if not text:
            return jsonify({"error": "no-text-provided"}), 400

        app.logger.info("TTS /speak called, model=%s voice=%s text_len=%d", model, voice, len(text))

        payload = {
            "model": model,
            "input": text,
            "voice": voice,
            "format": "mp3"
        }

        resp = requests.post(
            "https://api.openai.com/v1/audio/speech",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60,
        )

        app.logger.info("TTS API status: %s", resp.status_code)

        if resp.status_code >= 400:
            try:
                err_json = resp.json()
            except ValueError:
                err_json = {"error": "non-json-response", "status_text": resp.text}
            app.logger.error("TTS API error: %s", err_json)
            return jsonify({"error": "openai_error", "details": err_json}), resp.status_code

        audio_bytes = resp.content
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

        return jsonify({"audio": audio_b64, "format": "mp3"})
    except Exception as e:
        app.logger.exception("speak error")
        return jsonify({"error": "proxy-error", "details": str(e)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)