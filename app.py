import os
import base64
import io
import time
from dotenv import load_dotenv
from flask import Flask, request, jsonify
import requests
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import stripe

load_dotenv()

# Optional Postgres for user/usage tracking (skip if env not set)
def _get_pg_conn():
    if not all([os.getenv("DATABASE_HOST"), os.getenv("DATABASE_USER"), os.getenv("DATABASE_NAME")]):
        return None
    try:
        import psycopg2
        return psycopg2.connect(
            host=os.getenv("DATABASE_HOST"),
            database=os.getenv("DATABASE_NAME"),
            user=os.getenv("DATABASE_USER"),
            password=os.getenv("DATABASE_PASSWORD", ""),
            port=os.getenv("DATABASE_PORT", "5432"),
            sslmode="require" if os.getenv("DATABASE_SSL", "true").lower() != "false" else "disable",
        )
    except Exception as e:
        import logging
        logging.warning("Postgres connect failed (tracking disabled): %s", e)
        return None


# Credits per event (database stores event_type; we sum credits in SQL)
CREDITS_REPLY = 40          # GENERATE_REPLY, SUMMARY, GENERATE_REPLY_MULTI
CREDITS_PLAY_VOICE = 50
CREDITS_RECORD_VOICE = 50
FREE_VOICE_PLAY_LIMIT = 2
FREE_VOICE_RECORD_LIMIT = 2


def _credits_sql():
    """SQL expression: sum of credits for events this UTC month."""
    return """
        COALESCE(SUM(CASE event_type
            WHEN 'GENERATE_REPLY' THEN %s
            WHEN 'SUMMARY' THEN %s
            WHEN 'GENERATE_REPLY_MULTI' THEN %s
            WHEN 'PLAY_VOICE' THEN %s
            WHEN 'RECORD_VOICE' THEN %s
            ELSE 0 END), 0)
    """ % (CREDITS_REPLY, CREDITS_REPLY, CREDITS_REPLY, CREDITS_PLAY_VOICE, CREDITS_RECORD_VOICE)


def _is_developer_email(email):
    """True if email is in DEVELOPER_EMAILS env (comma-separated). Never blocks dev/test accounts."""
    if not email or not isinstance(email, str):
        return False
    dev_list = os.getenv("DEVELOPER_EMAILS", "").strip()
    if not dev_list:
        return False
    emails = [e.strip().lower() for e in dev_list.split(",") if e.strip()]
    return email.strip().lower() in emails


def check_quota(user_id, required_credits=0, for_voice_play=False, for_voice_record=False, user_email=None):
    """
    Check if user is within monthly quota (credit-based) and optional voice limits.
    Returns (allowed, used_credits, quota, voice_plays_used, voice_records_used).
    If DEVELOPER_EMAILS contains user_email, always allowed (no block for dev/test).
    If no DB or user not found, returns (True, 0, 1000, 0, 0).
    """
    if user_email and _is_developer_email(user_email):
        return True, 0, 1000, 0, 0
    conn = _get_pg_conn()
    if not conn or not user_id:
        return True, 0, 1000, 0, 0
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT monthly_quota, plan, is_blocked FROM users WHERE id = %s",
                (user_id,),
            )
            row = cur.fetchone()
        if not row:
            return True, 0, 1000, 0, 0
        monthly_quota, plan, is_blocked = row
        quota = int(monthly_quota or 1000)
        if is_blocked:
            return False, 0, quota, 0, 0
        is_paid = plan and plan.lower() in ("vip", "paid", "pro")
        month_start = "date_trunc('month', NOW() AT TIME ZONE 'UTC')"
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT """ + _credits_sql().strip() + """
                FROM usage_events
                WHERE user_id = %s AND created_at >= """ + month_start + """
                """,
                (user_id,),
            )
            used_credits = int(cur.fetchone()[0] or 0)
            cur.execute(
                """
                SELECT COUNT(*) FROM usage_events
                WHERE user_id = %s AND event_type = 'PLAY_VOICE' AND created_at >= """ + month_start + """
                """,
                (user_id,),
            )
            voice_plays_used = int(cur.fetchone()[0] or 0)
            cur.execute(
                """
                SELECT COUNT(*) FROM usage_events
                WHERE user_id = %s AND event_type = 'RECORD_VOICE' AND created_at >= """ + month_start + """
                """,
                (user_id,),
            )
            voice_records_used = int(cur.fetchone()[0] or 0)
        # Unlimited quota (e.g. -1)
        if quota < 0:
            allowed_quota = True
        else:
            allowed_quota = (used_credits + required_credits) <= quota
        allowed_play = is_paid or voice_plays_used < FREE_VOICE_PLAY_LIMIT
        allowed_record = is_paid or voice_records_used < FREE_VOICE_RECORD_LIMIT
        if for_voice_play:
            allowed = allowed_quota and allowed_play
        elif for_voice_record:
            allowed = allowed_quota and allowed_record
        else:
            allowed = allowed_quota
        return allowed, used_credits, quota, voice_plays_used, voice_records_used
    except Exception as e:
        app.logger.warning("Quota check failed: %s", e)
        return True, 0, 1000, 0, 0
    finally:
        try:
            conn.close()
        except Exception:
            pass


def track_user_and_usage(data):
    """If request has user info, upsert users and insert usage_events. Failures are logged only.
    Uses plan_ends_at when the column exists (after migration); otherwise falls back to basic upsert.
    """
    if not data or not isinstance(data.get("user"), dict):
        return
    user = data["user"]
    if not user.get("id") or not user.get("email"):
        return
    conn = _get_pg_conn()
    if not conn:
        return
    user_args = (
        user["id"],
        user["email"],
        user.get("provider") or "google",
        user.get("registeredAt") or None,
        user.get("lastLoginAt") or None,
    )
    usage_args = (
        user["id"],
        data.get("event_type") or "GENERATE_REPLY",
        data.get("model") or None,
    )
    try:
        with conn.cursor() as cur:
            end_of_month_sql = "(date_trunc('month', NOW() AT TIME ZONE 'UTC') + interval '1 month' - interval '1 second') AT TIME ZONE 'UTC'"
            # Prefer full upsert with plan_ends_at (roadmap: plan end, days remaining in view).
            try:
                cur.execute(
                    """
                    INSERT INTO users (id, email, provider, registered_at, last_login_at, updated_at, plan_ends_at)
                    VALUES (%s, %s, %s, %s::timestamptz, %s::timestamptz, NOW(), """ + end_of_month_sql + """)
                    ON CONFLICT (id) DO UPDATE SET
                        email = EXCLUDED.email,
                        provider = EXCLUDED.provider,
                        last_login_at = EXCLUDED.last_login_at,
                        updated_at = NOW(),
                        plan_ends_at = CASE
                            WHEN COALESCE(trim(lower(users.plan)), 'free') = 'free'
                            THEN """ + end_of_month_sql + """
                            ELSE users.plan_ends_at
                        END
                    """,
                    user_args,
                )
                # Also set plan_expires_at if column exists (some DBs/UI use this name).
                try:
                    cur.execute(
                        """
                        UPDATE users SET plan_expires_at = """ + end_of_month_sql + """
                        WHERE id::text = %s AND (plan IS NULL OR lower(trim(COALESCE(plan, ''))) = 'free')
                        """,
                        (user["id"],),
                    )
                except Exception:
                    pass
            except Exception as col_err:
                # Column plan_ends_at may not exist yet; fall back to basic upsert.
                err_code = getattr(col_err, "pgcode", None)
                if err_code == "42703":
                    cur.execute(
                        """
                        INSERT INTO users (id, email, provider, registered_at, last_login_at, updated_at)
                        VALUES (%s, %s, %s, %s::timestamptz, %s::timestamptz, NOW())
                        ON CONFLICT (id) DO UPDATE SET
                            email = EXCLUDED.email,
                            provider = EXCLUDED.provider,
                            last_login_at = EXCLUDED.last_login_at,
                            updated_at = NOW()
                        """,
                        user_args,
                    )
                    # If table has plan_expires_at instead, set it to end of month for free plan
                    try:
                        cur.execute(
                            """
                            UPDATE users SET plan_expires_at = (date_trunc('month', NOW() AT TIME ZONE 'UTC') + interval '1 month' - interval '1 second') AT TIME ZONE 'UTC'
                            WHERE id::text = %s AND (plan IS NULL OR lower(trim(COALESCE(plan, ''))) = 'free')
                            """,
                            (user["id"],),
                        )
                    except Exception:
                        pass
                else:
                    raise
            cur.execute(
                """
                INSERT INTO usage_events (user_id, event_type, model, created_at)
                VALUES (%s, %s, %s, NOW())
                """,
                usage_args,
            )
            # Keep used_credits_current_month in sync (no trigger required)
            try:
                cur.execute(
                    """
                    UPDATE users SET used_credits_current_month = COALESCE((
                        SELECT SUM(CASE ev.event_type
                            WHEN 'GENERATE_REPLY' THEN 40
                            WHEN 'SUMMARY' THEN 40
                            WHEN 'GENERATE_REPLY_MULTI' THEN 40
                            WHEN 'PLAY_VOICE' THEN 50
                            WHEN 'RECORD_VOICE' THEN 50
                            ELSE 0
                        END)::int
                        FROM usage_events ev
                        WHERE ev.user_id::text = users.id::text
                          AND ev.created_at >= date_trunc('month', NOW() AT TIME ZONE 'UTC')
                    ), 0)
                    WHERE users.id::text = %s
                    """,
                    (user["id"],),
                )
            except Exception as upd_err:
                if getattr(upd_err, "pgcode", None) != "42703":
                    raise
        conn.commit()
    except Exception as e:
        app.logger.exception("Track user/usage error: %s", e)
        try:
            conn.rollback()
        except Exception:
            pass
    finally:
        try:
            conn.close()
        except Exception:
            pass

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

# Stripe configuration (secret key from environment; DO NOT hardcode)
stripe.api_key = os.getenv("STRIPE_SECRET_KEY") or ""
STRIPE_PUBLISHABLE_KEY = os.getenv("STRIPE_PUBLISHABLE_KEY") or ""

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


@app.route("/create-checkout-session", methods=["POST"])
@limiter.limit("10 per minute")
def create_checkout_session():
    """
    Create a Stripe Checkout session for Cursor Pro.
    Expects JSON body: { customerEmail, billingMode: 'monthly'|'annual', customAmountUsd? }
    """
    try:
        if not stripe.api_key:
            return jsonify({"error": "stripe_not_configured"}), 500

        data = request.get_json(force=True) or {}
        email = data.get("customerEmail")
        billing_mode = data.get("billingMode") or "monthly"
        custom_amount = data.get("customAmountUsd")

        # Decide amount in USD
        monthly_price = 10.0
        annual_monthly_price = 7.0
        annual_total = annual_monthly_price * 12.0  # 84

        try:
            if custom_amount is not None and str(custom_amount).strip() != "":
                amount_usd = float(custom_amount)
            elif billing_mode == "annual":
                amount_usd = annual_total
            else:
                amount_usd = monthly_price
        except Exception:
            amount_usd = monthly_price

        if amount_usd < 5.0:
            return jsonify({"error": "amount_too_low", "message": "Minimum amount is $5"}), 400

        interval = "year" if billing_mode == "annual" else "month"

        session = stripe.checkout.Session.create(
            mode="subscription",
            customer_email=email or None,
            line_items=[
                {
                    "price_data": {
                        "currency": "usd",
                        "product_data": {"name": "Cursor Pro"},
                        "unit_amount": int(round(amount_usd * 100)),
                        "recurring": {"interval": interval},
                    },
                    "quantity": 1,
                }
            ],
            success_url=os.getenv("STRIPE_SUCCESS_URL", "https://example.com/stripe-success")
            + "?session_id={CHECKOUT_SESSION_ID}",
            cancel_url=os.getenv("STRIPE_CANCEL_URL", "https://example.com/stripe-cancel"),
        )

        return jsonify({"url": session.url})
    except Exception as e:
        app.logger.exception("Stripe checkout error")
        return jsonify({"error": "stripe_error", "message": str(e)}), 500


@app.route("/generate", methods=["POST"])
@limiter.limit("60 per minute")
def generate():
    """Generate email reply using GPT / DeepSeek / Gemini based on requested model"""
    try:
        auth_error = check_auth()
        if auth_error:
            return auth_error

        data = request.get_json() or {}
        user = data.get("user") if isinstance(data.get("user"), dict) else None
        user_id = user.get("id") if user else None
        used_credits, quota = 0, 1000

        # Quota check: 40 credits per text/summary reply. Do not count this request if over quota.
        if user_id:
            allowed, used_credits, quota, _vp, _vr = check_quota(
                user_id, required_credits=CREDITS_REPLY, user_email=user.get("email")
            )
            if not allowed:
                return jsonify({
                    "error": "quota_exceeded",
                    "message": "You've reached your free monthly limit. Upgrade to continue.",
                    "used": used_credits,
                    "quota": quota,
                }), 200

        email_content = data.get("emailContent", "")
        raw_model = (data.get("model") or "").strip()
        if raw_model in ("", "auto", None):
            raw_model = os.getenv("DEFAULT_MODEL", "gpt-3.5-turbo")

        # Map UI names to actual API model IDs (e.g. OpenAI reasoning models)
        _model_alias = {
            "o3": "o1",
            "o4-mini": "o1-mini",
        }
        model = _model_alias.get(raw_model, raw_model)
        # Gemini: 1.5 names often 404 on v1beta; use 2.0-flash as supported default
        if model.lower().startswith("gemini"):
            _gemini_alias = {"gemini-1.5-flash": "gemini-2.0-flash", "gemini-1.5-pro": "gemini-2.0-flash", "gemini-1.5-flash-latest": "gemini-2.0-flash", "gemini-1.5-pro-latest": "gemini-2.0-flash"}
            model = _gemini_alias.get(model.lower(), model)
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
        def call_chat_completions(base_url: str, api_key: str, prov: str):
            payload = {
                "model": model,
                "messages": messages,
                "max_tokens": data.get("max_tokens", 400),
            }
            n_val = int(data.get("n")) if data.get("n") else (2 if data.get("multi") else 1)
            # DeepSeek only supports n=1
            if prov == "deepseek" and n_val > 1:
                n_val = 1
            if n_val > 1:
                payload["n"] = n_val

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
            resp, err = call_chat_completions(base_url, api_key, provider)
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

        # Record usage only on successful reply (40 credits per text/summary)
        if user_id and (reply or replies):
            track_user_and_usage(data)
            used_credits += CREDITS_REPLY

        out = {"raw": result, "reply": reply, "replies": replies, "provider": provider, "model": model}
        if user_id:
            out["used"] = used_credits
            out["quota"] = quota
        return jsonify(out)
        
    except Exception as e:
        app.logger.exception("proxy error in /generate")
        return jsonify({"error": "proxy-error", "details": str(e)}), 500


@app.route("/transcribe", methods=["POST"])
@limiter.limit("30 per minute")
def transcribe():
    """Transcribe audio using OpenAI Whisper API.
    Free plan: 50 credits per record, max 2 voice records per month. Check DB for quota/limits.
    Request: { "audio": "base64...", "model": "whisper-1", "user": { id, email, ... } }
    """
    try:
        if not OPENAI_API_KEY:
            return jsonify({"error": "openai-key-not-configured"}), 500

        auth_error = check_auth()
        if auth_error:
            return auth_error

        data = request.get_json() or {}
        user = data.get("user") if isinstance(data.get("user"), dict) else None
        user_id = user.get("id") if user else None
        audio_base64 = data.get("audio", "")
        model = data.get("model", "whisper-1")

        if not audio_base64:
            return jsonify({"error": "no-audio-provided"}), 400

        if user_id:
            allowed, used_credits, quota, voice_plays_used, voice_records_used = check_quota(
                user_id, required_credits=CREDITS_RECORD_VOICE, for_voice_record=True, user_email=user.get("email")
            )
            if not allowed:
                return jsonify({
                    "error": "voice_record_locked" if voice_records_used >= FREE_VOICE_RECORD_LIMIT else "quota_exceeded",
                    "message": "Upgrade your plan to unlock this feature.",
                    "used": used_credits,
                    "quota": quota,
                    "voice_records_used": voice_records_used,
                    "voice_records_limit": FREE_VOICE_RECORD_LIMIT,
                }), 200

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

        if user_id:
            track_user_and_usage({"user": user, "event_type": "RECORD_VOICE", "model": model})

        return jsonify({"text": text, "raw": result})
        
    except Exception as e:
        app.logger.exception("transcribe error")
        return jsonify({"error": "proxy-error", "details": str(e)}), 500


@app.route("/speak", methods=["POST"])
@limiter.limit("30 per minute")
def speak():
    """
    Text-to-speech using OpenAI audio/speech API.
    Free plan: 50 credits per play, max 2 voice plays per month. Check DB for quota/limits.
    Request JSON: { "text": "...", "model": "...", "voice": "...", "user": { id, email, ... } }
    """
    try:
        if not OPENAI_API_KEY:
            return jsonify({"error": "openai-key-not-configured"}), 500

        auth_error = check_auth()
        if auth_error:
            return auth_error

        data = request.get_json() or {}
        user = data.get("user") if isinstance(data.get("user"), dict) else None
        user_id = user.get("id") if user else None
        text = (data.get("text") or "").strip()
        model = data.get("model") or "gpt-4o-mini-tts"
        voice = data.get("voice") or "alloy"

        if not text:
            return jsonify({"error": "no-text-provided"}), 400

        if user_id:
            allowed, used_credits, quota, voice_plays_used, voice_records_used = check_quota(
                user_id, required_credits=CREDITS_PLAY_VOICE, for_voice_play=True, user_email=user.get("email")
            )
            if not allowed:
                return jsonify({
                    "error": "voice_play_locked" if voice_plays_used >= FREE_VOICE_PLAY_LIMIT else "quota_exceeded",
                    "message": "Upgrade your plan to unlock this feature.",
                    "used": used_credits,
                    "quota": quota,
                    "voice_plays_used": voice_plays_used,
                    "voice_plays_limit": FREE_VOICE_PLAY_LIMIT,
                }), 200

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

        if user_id:
            track_user_and_usage({"user": user, "event_type": "PLAY_VOICE", "model": model})

        return jsonify({"audio": audio_b64, "format": "mp3"})
    except Exception as e:
        app.logger.exception("speak error")
        return jsonify({"error": "proxy-error", "details": str(e)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)