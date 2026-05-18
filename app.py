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

# Monthly credit pools (each AI suggestion run ≈ 40 credits)
FREE_MONTHLY_QUOTA = 2000       # ~50 suggestion runs / month
TIER_5_MONTHLY_QUOTA = 6000     # ~150 suggestion runs / month ($5)
TIER_10_MONTHLY_QUOTA = 15000   # ~375 suggestion runs / month ($10)

FREE_VOICE_PLAY_LIMIT = 8
FREE_VOICE_RECORD_LIMIT = 8
TIER_5_VOICE_LIMIT = 75
TIER_10_VOICE_LIMIT = 200


def _voice_limits_from_monthly_quota(monthly_quota: int):
    """
    Map monthly_quota → voice play/record limits (separate from credit pool).
    """
    try:
        q = int(monthly_quota)
    except Exception:
        q = FREE_MONTHLY_QUOTA

    # Unlimited credits tier (if any)
    if q < 0:
        return TIER_10_VOICE_LIMIT, TIER_10_VOICE_LIMIT

    if q >= TIER_10_MONTHLY_QUOTA:
        return TIER_10_VOICE_LIMIT, TIER_10_VOICE_LIMIT
    if q >= TIER_5_MONTHLY_QUOTA:
        return TIER_5_VOICE_LIMIT, TIER_5_VOICE_LIMIT
    if q >= FREE_MONTHLY_QUOTA:
        return FREE_VOICE_PLAY_LIMIT, FREE_VOICE_RECORD_LIMIT
    return 2, 2


def _infer_monthly_quota_from_subscription_price_cents(unit_amount_cents: int | None, interval: str | None):
    """
    Map Stripe subscription price → monthly_quota (credits).
      - $5/month  -> TIER_5_MONTHLY_QUOTA
      - $10/month -> TIER_10_MONTHLY_QUOTA
    For yearly prices, we assume the monthly equivalent and map to the same tiers.
    """
    if not unit_amount_cents:
        return None

    interval_norm = (interval or "").strip().lower()
    monthly_equiv_usd = None

    try:
        usd = float(unit_amount_cents) / 100.0
        if interval_norm == "month":
            monthly_equiv_usd = usd
        elif interval_norm == "year":
            monthly_equiv_usd = usd / 12.0
    except Exception:
        return None

    if monthly_equiv_usd is None:
        return None

  # Tier mapping by monthly equivalent
  # Using thresholds keeps this robust to minor price formatting differences.
    if monthly_equiv_usd < 7.0:
        return TIER_5_MONTHLY_QUOTA
    return TIER_10_MONTHLY_QUOTA


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


def _has_generate_page_pro(user_email, monthly_quota, plan=None):
    """Pro visual email builder: paid tier, non-free plan, or developer allowlist."""
    if user_email and _is_developer_email(user_email):
        return True
    p = str(plan or "").strip().lower()
    if p and p not in ("free", ""):
        return True
    try:
        return int(monthly_quota or 0) >= TIER_5_MONTHLY_QUOTA
    except Exception:
        return False


def check_quota(user_id, required_credits=0, for_voice_play=False, for_voice_record=False, user_email=None):
    """
    Check if user is within monthly quota (credit-based) and optional voice limits.
    Returns (allowed, used_credits, quota, voice_plays_used, voice_records_used).
    If DEVELOPER_EMAILS contains user_email, always allowed (no block for dev/test).
    If no DB or user not found, returns (True, 0, FREE_MONTHLY_QUOTA, 0, 0).
    """
    if user_email and _is_developer_email(user_email):
        return True, 0, FREE_MONTHLY_QUOTA, 0, 0
    conn = _get_pg_conn()
    if not conn or not user_id:
        return True, 0, FREE_MONTHLY_QUOTA, 0, 0
    try:
        with conn.cursor() as cur:
            # Prefer including usage_reset_at (added later). Fall back if column doesn't exist.
            usage_reset_at = None
            row = None
            try:
                cur.execute(
                    "SELECT monthly_quota, plan, is_blocked, usage_reset_at FROM users WHERE id::text = %s::text",
                    (str(user_id),),
                )
                row = cur.fetchone()
                if row and len(row) >= 4:
                    usage_reset_at = row[3]
            except Exception as col_err:
                err_code = getattr(col_err, "pgcode", None)
                # Undefined column
                if err_code == "42703":
                    cur.execute(
                        "SELECT monthly_quota, plan, is_blocked FROM users WHERE id::text = %s::text",
                        (str(user_id),),
                    )
                    row = cur.fetchone()
                else:
                    raise
        if not row:
            return True, 0, FREE_MONTHLY_QUOTA, 0, 0
        monthly_quota, plan, is_blocked = row[0], row[1], row[2]
        quota = int(monthly_quota or FREE_MONTHLY_QUOTA)
        if is_blocked:
            return False, 0, quota, 0, 0
        # Compute voice limits from monthly_quota, not from "is_paid".
        voice_play_limit, voice_record_limit = _voice_limits_from_monthly_quota(int(monthly_quota or FREE_MONTHLY_QUOTA))
        month_start = "date_trunc('month', NOW() AT TIME ZONE 'UTC')"
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT """ + _credits_sql().strip() + """
                FROM usage_events
                WHERE user_id::text = %s::text AND created_at >= """ + month_start + """
                """,
                (str(user_id),),
            )
            used_credits_total = int(cur.fetchone()[0] or 0)

            # If user upgraded mid-month, count "free" usage only up to FREE_MONTHLY_QUOTA before upgrade,
            # then count usage after upgrade against paid quota.
            used_credits_effective = used_credits_total
            try:
                if usage_reset_at and str(plan or "").strip().lower() != "free":
                    # Only apply within the same UTC month.
                    cur.execute(
                        """
                        SELECT """ + _credits_sql().strip() + """
                        FROM usage_events
                        WHERE user_id::text = %s::text AND created_at >= """ + month_start + """ AND created_at < %s
                        """,
                        (str(user_id), usage_reset_at),
                    )
                    used_before = int(cur.fetchone()[0] or 0)
                    used_after = max(0, used_credits_total - used_before)
                    used_credits_effective = min(used_before, FREE_MONTHLY_QUOTA) + used_after
                    # Bonus: remaining free quota + paid quota (paid quota is users.monthly_quota)
                    quota = int(monthly_quota or FREE_MONTHLY_QUOTA) + FREE_MONTHLY_QUOTA
            except Exception:
                used_credits_effective = used_credits_total

            cur.execute(
                """
                SELECT COUNT(*) FROM usage_events
                WHERE user_id::text = %s::text AND event_type = 'PLAY_VOICE' AND created_at >= """ + month_start + """
                """,
                (str(user_id),),
            )
            voice_plays_used = int(cur.fetchone()[0] or 0)
            cur.execute(
                """
                SELECT COUNT(*) FROM usage_events
                WHERE user_id::text = %s::text AND event_type = 'RECORD_VOICE' AND created_at >= """ + month_start + """
                """,
                (str(user_id),),
            )
            voice_records_used = int(cur.fetchone()[0] or 0)
        # Unlimited quota (e.g. -1)
        if quota < 0:
            allowed_quota = True
        else:
            allowed_quota = (used_credits_effective + required_credits) <= quota
        allowed_play = voice_plays_used < voice_play_limit
        allowed_record = voice_records_used < voice_record_limit
        if for_voice_play:
            allowed = allowed_quota and allowed_play
        elif for_voice_record:
            allowed = allowed_quota and allowed_record
        else:
            allowed = allowed_quota
        return allowed, used_credits_effective, quota, voice_plays_used, voice_records_used
    except Exception as e:
        app.logger.warning("Quota check failed: %s", e)
        return True, 0, FREE_MONTHLY_QUOTA, 0, 0
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
    html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <meta name="google-site-verification" content="google10339bab4366663c" />
  <title>AI Email Assistance</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
      background: #f7f8fc;
      color: #1a1a2e;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }
    header {
      background: #1f3d53;
      color: #fff;
      padding: 18px 40px;
      display: flex;
      align-items: center;
      gap: 14px;
    }
    header h1 { font-size: 22px; font-weight: 700; letter-spacing: -0.3px; }
    header span { font-size: 13px; opacity: 0.7; margin-left: auto; }
    main {
      flex: 1;
      max-width: 780px;
      margin: 60px auto;
      padding: 0 24px;
      text-align: center;
    }
    .badge {
      display: inline-block;
      background: #e8f4fd;
      color: #1f3d53;
      font-size: 12px;
      font-weight: 600;
      letter-spacing: 0.8px;
      text-transform: uppercase;
      padding: 5px 14px;
      border-radius: 20px;
      margin-bottom: 22px;
    }
    h2 {
      font-size: 38px;
      font-weight: 800;
      line-height: 1.2;
      margin-bottom: 18px;
      color: #1a1a2e;
    }
    .subtitle {
      font-size: 18px;
      color: #555;
      line-height: 1.7;
      max-width: 600px;
      margin: 0 auto 40px;
    }
    .features {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
      gap: 20px;
      margin: 0 auto 50px;
      text-align: left;
    }
    .feature {
      background: #fff;
      border: 1px solid #e5e7eb;
      border-radius: 12px;
      padding: 22px 20px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    .feature .icon { font-size: 26px; margin-bottom: 10px; }
    .feature h3 { font-size: 15px; font-weight: 700; margin-bottom: 6px; color: #1f3d53; }
    .feature p { font-size: 13px; color: #666; line-height: 1.6; }
    .cta {
      background: #1f3d53;
      color: #fff;
      border: none;
      padding: 14px 36px;
      border-radius: 8px;
      font-size: 16px;
      font-weight: 600;
      cursor: pointer;
      text-decoration: none;
      display: inline-block;
      margin-bottom: 16px;
    }
    .cta:hover { background: #16303f; }
    .note { font-size: 13px; color: #888; margin-bottom: 50px; }
    footer {
      background: #fff;
      border-top: 1px solid #e5e7eb;
      text-align: center;
      padding: 20px;
      font-size: 13px;
      color: #888;
    }
    footer a { color: #1f3d53; text-decoration: none; margin: 0 10px; font-weight: 500; }
    footer a:hover { text-decoration: underline; }
  </style>
</head>
<body>
  <header>
    <h1>AI Email Assistance</h1>
    <span>Chrome Extension</span>
  </header>

  <main>
    <div class="badge">Gmail Chrome Extension</div>
    <h2>Write better emails,<br/>faster &mdash; with AI</h2>
    <p class="subtitle">
      AI Email Assistance is a Chrome extension that integrates directly into Gmail.
      It uses artificial intelligence to help you compose, reply to, and refine emails
      &mdash; saving time and improving the quality of every message you send.
    </p>

    <div class="features">
      <div class="feature">
        <div class="icon">&#9889;</div>
        <h3>AI Reply Suggestions</h3>
        <p>Instantly generate smart, context-aware email replies with one click directly inside Gmail.</p>
      </div>
      <div class="feature">
        <div class="icon">&#9997;&#65039;</div>
        <h3>Email Composition</h3>
        <p>Describe what you want to say and let AI draft a polished email for you in seconds.</p>
      </div>
      <div class="feature">
        <div class="icon">&#127908;</div>
        <h3>Voice-to-Email</h3>
        <p>Record your voice and have AI transcribe and format it into a professional email message.</p>
      </div>
      <div class="feature">
        <div class="icon">&#127757;</div>
        <h3>Multi-language</h3>
        <p>Generate and refine emails in multiple languages to communicate globally with confidence.</p>
      </div>
    </div>

    <a href="/pricing" class="cta">View Plans &amp; Pricing</a>
    <p class="note">Install from the Chrome Web Store &middot; Works inside Gmail</p>

    <div style="margin-top:40px;padding:24px 28px;background:#fff;border:1px solid #e5e7eb;border-radius:12px;text-align:left;max-width:600px;margin-left:auto;margin-right:auto;">
      <h3 style="font-size:15px;font-weight:700;color:#1f3d53;margin-bottom:10px;">About AI Email Assistance</h3>
      <p style="font-size:13px;color:#555;line-height:1.7;margin-bottom:14px;">
        <strong>AI Email Assistance</strong> is a Chrome extension that connects to Gmail to help users write, reply to,
        and improve emails using artificial intelligence. The extension accesses your Gmail interface to read the current
        email thread and generate contextually relevant replies and compositions. No email content is stored on our servers.
      </p>
      <p style="font-size:13px;color:#555;line-height:1.7;">
        By using this extension you agree to our
        <a href="/terms" style="color:#1f3d53;font-weight:600;">Terms of Service</a>.
        For information on how we handle your data, please read our
        <a href="/privacy" style="color:#1f3d53;font-weight:600;">Privacy Policy</a>.
      </p>
    </div>
  </main>

  <footer>
    <p>
      &copy; 2026 AI Email Assistance &nbsp;&middot;&nbsp;
      <a href="/privacy">Privacy Policy</a>
      <a href="/terms">Terms of Service</a>
      <a href="/pricing">Pricing</a>
    </p>
  </footer>
</body>
</html>"""
    return html, 200, {'Content-Type': 'text/html; charset=utf-8'}


@app.route("/google10339bab4366663c.html", methods=["GET"])
def google_site_verification():
    return "google-site-verification: google10339bab4366663c.html", 200, {'Content-Type': 'text/html; charset=utf-8'}


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


def _set_user_plan_by_email(
    email: str,
    plan: str = "pro",
    customer_id: str | None = None,
    plan_ends_at_ts: int | None = None,
    monthly_quota: int | None = None,
):
    """Update users table to mark user as paid/pro. No-op if DB not configured."""
    if not email:
        return
    conn = _get_pg_conn()
    if not conn:
        return
    try:
        with conn.cursor() as cur:
            # Detect whether this is an upgrade from free → apply bonus only once.
            apply_bonus = False
            try:
                cur.execute(
                    "SELECT plan FROM users WHERE lower(email) = lower(%s) LIMIT 1",
                    (email,),
                )
                row = cur.fetchone()
                prev_plan = (row[0] if row else None) or "free"
                prev_plan = str(prev_plan).strip().lower()
                apply_bonus = (prev_plan == "" or prev_plan == "free")
            except Exception:
                # If we cannot read previous plan (or user row doesn't exist yet), apply bonus.
                apply_bonus = True
            # When upgrading from free, set usage_reset_at if the column exists.
            # This enables "remaining free quota + paid quota" accounting in check_quota.
            if apply_bonus:
                try:
                    cur.execute(
                        "UPDATE users SET usage_reset_at = NOW() WHERE lower(email) = lower(%s)",
                        (email,),
                    )
                except Exception:
                    pass

            effective_quota = monthly_quota

            if plan_ends_at_ts:
                cur.execute(
                    """
                    UPDATE users
                    SET plan = %s,
                        is_blocked = FALSE,
                        payment_customer_id = COALESCE(%s, payment_customer_id),
                        plan_ends_at = to_timestamp(%s),
                        plan_expires_at = to_timestamp(%s),
                        monthly_quota = COALESCE(%s, monthly_quota),
                        updated_at = NOW()
                    WHERE lower(email) = lower(%s)
                    """,
                    (plan, customer_id, int(plan_ends_at_ts), int(plan_ends_at_ts), effective_quota, email),
                )
            else:
                cur.execute(
                    """
                    UPDATE users
                    SET plan = %s,
                        is_blocked = FALSE,
                        payment_customer_id = COALESCE(%s, payment_customer_id),
                        monthly_quota = COALESCE(%s, monthly_quota),
                        updated_at = NOW()
                    WHERE lower(email) = lower(%s)
                    """,
                    (plan, customer_id, effective_quota, email),
                )
            conn.commit()
    except Exception as e:
        app.logger.warning("Failed to set user plan: %s", e)
        try:
            conn.rollback()
        except Exception:
            pass
    finally:
        try:
            conn.close()
        except Exception:
            pass


@app.route("/stripe-webhook", methods=["POST"])
def stripe_webhook():
    """
    Stripe webhook endpoint. Configure STRIPE_WEBHOOK_SECRET in env and add this URL in Stripe dashboard.
    On successful payment, mark user as pro in DB (users.plan = 'pro').
    """
    webhook_secret = os.getenv("STRIPE_WEBHOOK_SECRET") or ""
    if not webhook_secret:
        return jsonify({"error": "webhook_not_configured"}), 500

    payload = request.data
    sig_header = request.headers.get("Stripe-Signature")
    try:
        event = stripe.Webhook.construct_event(payload=payload, sig_header=sig_header, secret=webhook_secret)
    except Exception as e:
        app.logger.warning("Stripe webhook signature verification failed: %s", e)
        return jsonify({"error": "invalid_signature"}), 400

    event_type = (event.get("type") or "").strip()
    obj = event.get("data", {}).get("object", {}) if isinstance(event.get("data"), dict) else {}

    try:
        # Checkout completed (subscription created)
        if event_type == "checkout.session.completed":
            email = obj.get("customer_details", {}).get("email") or obj.get("customer_email")
            customer_id = obj.get("customer")
            # If subscription exists, fetch period end to set plan end date
            sub_id = obj.get("subscription")
            plan_end_ts = None
            monthly_quota = None
            try:
                if sub_id and stripe.api_key:
                    # Expand price so we can infer whether this is $5/month or $10/month.
                    sub = stripe.Subscription.retrieve(sub_id, expand=["items.data.price"])
                    plan_end_ts = sub.get("current_period_end")
                    items = (sub.get("items") or {}).get("data") or []
                    if items:
                        price = (items[0] or {}).get("price") or {}
                        unit_amount = price.get("unit_amount")
                        recurring = price.get("recurring") or {}
                        interval = recurring.get("interval")
                        monthly_quota = _infer_monthly_quota_from_subscription_price_cents(unit_amount, interval)
            except Exception:
                plan_end_ts = None
            # Fallback: best-effort inference from total amount (cents).
            if monthly_quota is None:
                try:
                    amount_total_cents = obj.get("amount_total")
                    if amount_total_cents:
                        amount_usd = float(amount_total_cents) / 100.0
                        # $5/mo is ~5, $10/mo is ~10, annual plans are much higher but still map to 6000.
                        monthly_quota = TIER_5_MONTHLY_QUOTA if amount_usd <= 6.0 else TIER_10_MONTHLY_QUOTA
                except Exception:
                    monthly_quota = None
            _set_user_plan_by_email(
                email=email,
                plan="pro",
                customer_id=customer_id,
                plan_ends_at_ts=plan_end_ts,
                monthly_quota=monthly_quota if monthly_quota is not None else TIER_10_MONTHLY_QUOTA,
            )

        # Subscription renewed / paid invoice
        # Stripe commonly emits: "invoice.paid" and/or "invoice.payment_succeeded"
        # Some dashboards/tools may show "invoice_payment.paid" naming — treat them equivalently.
        elif event_type in ("invoice.paid", "invoice.payment_succeeded", "invoice_payment.paid"):
            # NOTE: Stripe Workbench may send invoice_payment objects here:
            # { object: 'invoice_payment', invoice: 'in_...', ... }
            # In that case we must fetch the invoice to get customer/subscription/email.
            email = obj.get("customer_email")
            customer_id = obj.get("customer")
            sub_id = obj.get("subscription")
            plan_end_ts = None
            monthly_quota = None
            try:
                if stripe.api_key:
                    inv_id = obj.get("invoice")
                    if (not email or not customer_id or not sub_id) and inv_id:
                        try:
                            inv = stripe.Invoice.retrieve(inv_id, expand=["customer", "subscription"])
                            if not customer_id:
                                cust = inv.get("customer")
                                customer_id = cust.get("id") if isinstance(cust, dict) else cust
                            if not email:
                                cust2 = inv.get("customer")
                                if isinstance(cust2, dict) and cust2.get("email"):
                                    email = cust2.get("email")
                                else:
                                    email = inv.get("customer_email") or inv.get("customer_details", {}).get("email")
                            if not sub_id:
                                sub = inv.get("subscription")
                                sub_id = sub.get("id") if isinstance(sub, dict) else sub
                        except Exception:
                            pass

                    if sub_id:
                        sub = stripe.Subscription.retrieve(sub_id, expand=["items.data.price"])
                        plan_end_ts = sub.get("current_period_end")
                        items = (sub.get("items") or {}).get("data") or []
                        if items:
                            price = (items[0] or {}).get("price") or {}
                            unit_amount = price.get("unit_amount")
                            recurring = price.get("recurring") or {}
                            interval = recurring.get("interval")
                            monthly_quota = _infer_monthly_quota_from_subscription_price_cents(unit_amount, interval)
            except Exception:
                plan_end_ts = None
            if monthly_quota is None:
                try:
                    # Invoice totals are in cents; we only care about detecting ~$5 vs ~$10 tiers.
                    amount_total_cents = obj.get("amount_paid") or obj.get("amount_due") or obj.get("total")
                    if amount_total_cents:
                        amount_usd = float(amount_total_cents) / 100.0
                        monthly_quota = TIER_5_MONTHLY_QUOTA if amount_usd <= 6.0 else TIER_10_MONTHLY_QUOTA
                except Exception:
                    monthly_quota = None
            _set_user_plan_by_email(
                email=email,
                plan="pro",
                customer_id=customer_id,
                plan_ends_at_ts=plan_end_ts,
                monthly_quota=monthly_quota if monthly_quota is not None else TIER_10_MONTHLY_QUOTA,
            )

        # Subscription cancelled → downgrade to free
        # NOTE: Stripe may emit either:
        # - customer.subscription.updated (when cancel_at_period_end is set)
        # - customer.subscription.deleted (when canceled immediately / ended)
        #
        # If cancel_at_period_end=True, the user should generally remain "pro" until current_period_end,
        # but we should still persist plan_ends_at so the app can downgrade later if needed.
        elif event_type in ("customer.subscription.updated", "customer.subscription.deleted"):
            customer_id = obj.get("customer")
            status = (obj.get("status") or "").strip().lower()
            cancel_at_period_end = bool(obj.get("cancel_at_period_end"))
            current_period_end = obj.get("current_period_end")
            ended_at = obj.get("ended_at")

            def _update_user_by_customer_id(*, set_free_now: bool):
                conn = _get_pg_conn()
                if not conn or not customer_id:
                    return
                try:
                    with conn.cursor() as cur:
                        if set_free_now:
                            # Downgrade immediately.
                            try:
                                cur.execute(
                                    """
                                    UPDATE users
                                    SET plan = 'free',
                                        is_blocked = FALSE,
                                        monthly_quota = FREE_MONTHLY_QUOTA,
                                        plan_ends_at = NULL,
                                        plan_expires_at = NULL,
                                        updated_at = NOW()
                                    WHERE payment_customer_id = %s
                                    """,
                                    (customer_id,),
                                )
                            except Exception as col_err:
                                # If plan_ends_at/plan_expires_at columns don't exist, fall back.
                                if getattr(col_err, "pgcode", None) == "42703":
                                    cur.execute(
                                        """
                                        UPDATE users
                                        SET plan = 'free',
                                            is_blocked = FALSE,
                                            monthly_quota = FREE_MONTHLY_QUOTA,
                                            updated_at = NOW()
                                        WHERE payment_customer_id = %s
                                        """,
                                        (customer_id,),
                                    )
                                else:
                                    raise
                        else:
                            # Cancel-at-period-end: keep plan as-is, but persist end timestamp if possible.
                            if current_period_end:
                                try:
                                    cur.execute(
                                        """
                                        UPDATE users
                                        SET plan_ends_at = to_timestamp(%s),
                                            plan_expires_at = to_timestamp(%s),
                                            updated_at = NOW()
                                        WHERE payment_customer_id = %s
                                        """,
                                        (int(current_period_end), int(current_period_end), customer_id),
                                    )
                                except Exception as col_err:
                                    if getattr(col_err, "pgcode", None) == "42703":
                                        # Column missing: ignore quietly.
                                        pass
                                    else:
                                        raise
                        conn.commit()
                except Exception as e:
                    app.logger.warning("Failed to apply subscription cancel/update: %s", e)
                    try:
                        conn.rollback()
                    except Exception:
                        pass
                finally:
                    try:
                        conn.close()
                    except Exception:
                        pass

            # Determine whether to downgrade now or keep pro-until-end.
            downgrade_now = (
                event_type == "customer.subscription.deleted"
                or status == "canceled"
                or bool(ended_at)
                or (not cancel_at_period_end and status in ("canceled", "unpaid"))
            )
            if downgrade_now:
                _update_user_by_customer_id(set_free_now=True)
            elif cancel_at_period_end:
                _update_user_by_customer_id(set_free_now=False)

    except Exception as e:
        app.logger.exception("Stripe webhook handler error: %s", e)
        return jsonify({"error": "webhook_handler_error"}), 500

    return jsonify({"received": True}), 200


@app.route("/sync-plan", methods=["POST"])
@limiter.limit("30 per minute")
def sync_plan():
    """
    Manual plan sync helper (no webhook required).
    Use this when a user paid BEFORE the webhook endpoint existed.

    Body: { "email": "..."} OR { "user": { "email": "...", "id": "..." } }
    Returns: { plan, monthly_quota, customer_id, plan_ends_at }
    """
    try:
        auth_error = check_auth()
        if auth_error:
            return auth_error

        if not stripe.api_key:
            return jsonify({"error": "stripe_not_configured"}), 500

        data = request.get_json(force=True) or {}
        email = (data.get("email") or "").strip()
        user = data.get("user") if isinstance(data.get("user"), dict) else None
        if not email and user:
            email = (user.get("email") or "").strip()
        if not email:
            return jsonify({"error": "missing_email"}), 400

        # Find Stripe customer by email (best-effort).
        customer_id = None
        try:
            # Customer.search is supported on modern Stripe API versions
            res = stripe.Customer.search(query=f"email:'{email}'", limit=1)
            if res and res.get("data"):
                customer_id = (res["data"][0] or {}).get("id")
        except Exception:
            try:
                # Fallback: list + filter
                res2 = stripe.Customer.list(email=email, limit=1)
                if res2 and res2.get("data"):
                    customer_id = (res2["data"][0] or {}).get("id")
            except Exception:
                customer_id = None

        if not customer_id:
            return jsonify({"error": "stripe_customer_not_found"}), 200

        # Get the most relevant active subscription.
        sub = None
        try:
            subs = stripe.Subscription.list(customer=customer_id, status="all", limit=10, expand=["data.items.data.price"])
            for s in (subs.get("data") or []):
                st = (s.get("status") or "").lower()
                if st in ("active", "trialing"):
                    sub = s
                    break
            if not sub and (subs.get("data") or []):
                sub = (subs.get("data") or [None])[0]
        except Exception:
            sub = None

        if not sub:
            return jsonify({"error": "stripe_subscription_not_found", "customer_id": customer_id}), 200

        plan_end_ts = sub.get("current_period_end")
        monthly_quota = None
        try:
            items = (sub.get("items") or {}).get("data") or []
            if items:
                price = (items[0] or {}).get("price") or {}
                unit_amount = price.get("unit_amount")
                recurring = price.get("recurring") or {}
                interval = recurring.get("interval")
                monthly_quota = _infer_monthly_quota_from_subscription_price_cents(unit_amount, interval)
        except Exception:
            monthly_quota = None
        if monthly_quota is None:
            monthly_quota = TIER_10_MONTHLY_QUOTA

        _set_user_plan_by_email(
            email=email,
            plan="pro",
            customer_id=customer_id,
            plan_ends_at_ts=plan_end_ts,
            monthly_quota=monthly_quota,
        )

        return jsonify({
            "ok": True,
            "email": email,
            "plan": "pro",
            "monthly_quota": monthly_quota,
            "customer_id": customer_id,
            "plan_ends_at": plan_end_ts,
        }), 200
    except Exception as e:
        app.logger.exception("sync-plan error")
        return jsonify({"error": "sync_failed", "details": str(e)}), 500


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
        used_credits, quota = 0, FREE_MONTHLY_QUOTA

        event_type = (data.get("event_type") or "").strip().upper()
        is_assistance = (event_type == "ASSISTANCE")
        required_credits = 0 if is_assistance else CREDITS_REPLY

        # Quota check (AI Assistance is free/no-credit by design).
        if user_id:
            allowed, used_credits, quota, _vp, _vr = check_quota(
                user_id, required_credits=required_credits, user_email=user.get("email")
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
            user_email = (user.get("email") if isinstance(user, dict) else None) or None
            user_plan = None
            try:
                conn_gp = _get_pg_conn()
                if conn_gp and user_id:
                    with conn_gp.cursor() as cur_gp:
                        cur_gp.execute(
                            "SELECT plan FROM users WHERE id::text = %s::text",
                            (str(user_id),),
                        )
                        row_gp = cur_gp.fetchone()
                        if row_gp:
                            user_plan = row_gp[0]
            except Exception:
                user_plan = None
            out["generate_page_pro"] = _has_generate_page_pro(user_email, quota, user_plan)
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
        # Let Whisper auto-detect language unless the client explicitly requests one.
        language = (data.get("language") or "").strip()
        prompt = (data.get("prompt") or "").strip()

        if not audio_base64:
            return jsonify({"error": "no-audio-provided"}), 400

        if user_id:
            allowed, used_credits, quota, voice_plays_used, voice_records_used = check_quota(
                user_id, required_credits=CREDITS_RECORD_VOICE, for_voice_record=True, user_email=user.get("email")
            )
            if not allowed:
                _, voice_record_limit = _voice_limits_from_monthly_quota(quota)
                return jsonify({
                    "error": "voice_record_locked" if voice_records_used >= voice_record_limit else "quota_exceeded",
                    "message": "Upgrade your plan to unlock this feature.",
                    "used": used_credits,
                    "quota": quota,
                    "voice_records_used": voice_records_used,
                    "voice_records_limit": voice_record_limit,
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
        # Low temperature reduces hallucinations; language is optional (auto-detect if omitted)
        data_payload = {"model": model, "temperature": "0"}
        if language:
            data_payload["language"] = language
        if prompt:
            data_payload["prompt"] = prompt

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
                voice_play_limit, _voice_record_limit = _voice_limits_from_monthly_quota(quota)
                return jsonify({
                    "error": "voice_play_locked" if voice_plays_used >= voice_play_limit else "quota_exceeded",
                    "message": "Upgrade your plan to unlock this feature.",
                    "used": used_credits,
                    "quota": quota,
                    "voice_plays_used": voice_plays_used,
                    "voice_plays_limit": voice_play_limit,
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


@app.route("/pricing", methods=["GET"])
def pricing_page():
    """Hosted Stripe pricing table page for extension upgrade flow."""
    pricing_table_id = os.getenv("STRIPE_PRICING_TABLE_ID") or "prctbl_1TR53pAWmPYcgBYulgQNg59b"
    pub_key = STRIPE_PUBLISHABLE_KEY or os.getenv("STRIPE_PUBLISHABLE_KEY") or ""
    if not pub_key:
        # Keep page functional but show a clear message if env isn't set.
        return """
<!doctype html><html><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>AI Email Assistance Pricing</title>
<style>body{font-family:Arial,sans-serif;margin:0;background:#f7f8fb;color:#111}.wrap{max-width:900px;margin:32px auto;padding:0 16px}h1{margin:0 0 8px;color:#092541}p{margin:0 0 20px;color:#555}</style>
</head><body><div class="wrap"><h1>Pricing unavailable</h1><p>Server is missing STRIPE_PUBLISHABLE_KEY. Set it in environment variables and redeploy.</p></div></body></html>
        """, 500, {"Content-Type": "text/html; charset=utf-8"}

    # IMPORTANT: do NOT use f-strings with raw CSS braces `{}` unless escaped.
    # Use str.format with escaped braces instead.
    html = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>AI Email Assistance Pricing</title>
  <script async src="https://js.stripe.com/v3/pricing-table.js"></script>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 0; background: #f7f8fb; color: #111; }}
    .wrap {{ max-width: 1100px; margin: 32px auto; padding: 0 16px; }}
    h1 {{ margin: 0 0 8px; color: #092541; }}
    p {{ margin: 0 0 20px; color: #555; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>AI Email Assistance Plans</h1>
    <p>Choose a plan to unlock higher credits and premium features.</p>
    <stripe-pricing-table
      pricing-table-id="{pricing_table_id}"
      publishable-key="{pub_key}">
    </stripe-pricing-table>
  </div>
</body>
</html>
    """
    return html.format(pricing_table_id=pricing_table_id, pub_key=pub_key), 200, {"Content-Type": "text/html; charset=utf-8"}


@app.route("/terms", methods=["GET"])
def terms_page():
    """Hosted Terms of Use page for extension footer links."""
    return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>AI Email Assistance - Terms of Use</title>
  <style>
    body { margin: 0; font-family: Arial, sans-serif; background: #f7f8fb; color: #1f2937; line-height: 1.62; }
    .wrap { max-width: 980px; margin: 28px auto; background: #fff; border: 1px solid #e5e7eb; border-radius: 12px; box-shadow: 0 8px 24px rgba(0,0,0,0.08); overflow: hidden; }
    .header { padding: 20px 24px; background: #092541; color: #fff; }
    .header h1 { margin: 0; font-size: 24px; font-weight: 700; }
    .header p { margin: 6px 0 0; font-size: 13px; color: rgba(255,255,255,0.86); }
    .content { padding: 24px; }
    h2 { margin: 22px 0 10px; font-size: 17px; color: #092541; }
    p { margin: 8px 0; font-size: 14px; }
    ul { margin: 8px 0 8px 20px; padding: 0; font-size: 14px; }
    li { margin: 6px 0; }
    .muted { color: #6b7280; font-size: 13px; }
    .contact { margin-top: 24px; padding: 12px 14px; background: #f3f7fc; border: 1px solid #d9e8fb; border-radius: 10px; }
    .contact a { color: #092541; font-weight: 700; text-decoration: none; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="header">
      <h1>AI Email Assistance - Terms of Use</h1>
      <p>Last updated: Apr 09, 2026</p>
    </div>
    <div class="content">
      <h2>AGREEMENT TO TERMS</h2>
      <p>These Terms of Use form a legally binding agreement between you (personally or on behalf of an entity) and AI Email Assistance ("Company", "we", "us", or "our"), regarding your access to and use of our extension, hosted pages, and related services (collectively, the "Service"). By accessing or using the Service, you confirm that you have read, understood, and agree to be bound by these Terms of Use.</p>
      <p>IF YOU DO NOT AGREE TO ALL OF THESE TERMS, YOU MUST NOT USE THE SERVICE.</p>
      <p>We may update these Terms from time to time for legal, security, or product reasons. Updated terms become effective when published on this page, and your continued use of the Service means you accept the updated terms.</p>
      <p>The Service is intended for users who are at least 13 years old. If you are under 18, you must use the Service under parent/guardian permission and supervision.</p>

      <h2>INTELLECTUAL PROPERTY RIGHTS</h2>
      <p>Unless stated otherwise, all software, design, UI, prompts, content, logos, features, and related materials in the Service are owned by us or licensed to us, and are protected by applicable intellectual property laws.</p>
      <p>We grant you a limited, non-exclusive, revocable, non-transferable license to access and use the Service for personal or internal business use only. You may not copy, sell, sublicense, reverse engineer, decompile, scrape, or exploit any part of the Service without our prior written consent.</p>

      <h2>USER REPRESENTATIONS</h2>
      <p>By using the Service, you represent and warrant that:</p>
      <ul>
        <li>information you provide is accurate and current;</li>
        <li>you have legal capacity to agree to these Terms;</li>
        <li>you will not use bots or automation to abuse the Service;</li>
        <li>you will not use the Service for unlawful, harmful, or deceptive activity;</li>
        <li>your usage will comply with all applicable laws and regulations.</li>
      </ul>

      <h2>ACCOUNT, SIGN-IN, AND SECURITY</h2>
      <p>Some features require Google sign-in. You are responsible for your account activity and for maintaining access security to your browser and email account. We may suspend or block accounts for abuse, fraud, or violations of these Terms.</p>

      <h2>PLANS, PAYMENTS, AND USAGE LIMITS</h2>
      <p>AI Email Assistance may offer free and paid plans. Credits, voice limits, and feature availability may vary by plan and may be updated over time.</p>
      <ul>
        <li>Free and paid tiers may include monthly quotas (e.g., AI credits, voice playback, and transcription limits).</li>
        <li>Usage limits may reset periodically (typically monthly UTC cycles).</li>
        <li>Payment processing is handled by third-party providers (e.g., Stripe), subject to their terms and policies.</li>
        <li>If payment is disputed, reversed, or refunded for abuse reasons, we may adjust plan access.</li>
      </ul>

      <h2>PROHIBITED USES AND LIMITATIONS</h2>
      <p>You agree not to use the Service to:</p>
      <ul>
        <li>send spam, phishing, scams, malware, or fraudulent content;</li>
        <li>generate illegal, hateful, threatening, or abusive communications;</li>
        <li>request or distribute sensitive credentials (passwords, one-time codes, banking PINs, etc.);</li>
        <li>impersonate others in a misleading or harmful way;</li>
        <li>bypass quotas, billing controls, feature locks, or security protections;</li>
        <li>attempt to disrupt, probe, or reverse engineer service infrastructure.</li>
      </ul>

      <h2>AI OUTPUT DISCLAIMER</h2>
      <p>AI-generated output may be incomplete, inaccurate, or unsuitable for your specific context. You are solely responsible for reviewing, editing, and approving all generated content before sending or relying on it.</p>

      <h2>PRIVACY AND DATA HANDLING</h2>
      <p>To operate the Service, we may process account identifiers, feature usage events, and request content needed to deliver AI functionality. By using the Service, you consent to this processing. Please avoid submitting highly sensitive or regulated data unless you are authorized and legally permitted to do so.</p>

      <h2>SUSPENSION AND TERMINATION</h2>
      <p>We may suspend, limit, or terminate access (with or without prior notice) for suspected abuse, security risk, legal compliance needs, non-payment, or violation of these Terms.</p>

      <h2>DISCLAIMERS AND LIMITATION OF LIABILITY</h2>
      <p>The Service is provided "as is" and "as available," without warranties of any kind, express or implied. To the maximum extent allowed by law, we are not liable for indirect, incidental, special, consequential, or punitive damages, or any loss of data, business, revenue, or reputation resulting from your use of the Service.</p>

      <h2>GOVERNING LAW</h2>
      <p>These Terms are governed by applicable laws in the jurisdiction where the Company operates, unless otherwise required by mandatory consumer law.</p>

      <h2>CONTACT US</h2>
      <div class="contact">
        <p>For legal notices, support, or Terms questions, contact:</p>
        <p><a href="mailto:aiemailassistance@gmail.com">aiemailassistance@gmail.com</a></p>
      </div>
      <p class="muted">By clicking "Connect with Google", you acknowledge and agree to these Terms of Use.</p>
    </div>
  </div>
</body>
</html>
    """, 200, {"Content-Type": "text/html; charset=utf-8"}


@app.route("/privacy", methods=["GET"])
def privacy_page():
    """Hosted Privacy Policy page for OAuth verification and extension footer links."""
    return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>AI Email Assistance - Privacy Policy</title>
  <style>
    body { margin: 0; font-family: Arial, sans-serif; background: #f7f8fb; color: #1f2937; line-height: 1.62; }
    .wrap { max-width: 980px; margin: 28px auto; background: #fff; border: 1px solid #e5e7eb; border-radius: 12px; box-shadow: 0 8px 24px rgba(0,0,0,0.08); overflow: hidden; }
    .header { padding: 20px 24px; background: #092541; color: #fff; }
    .header h1 { margin: 0; font-size: 24px; font-weight: 700; }
    .header p { margin: 6px 0 0; font-size: 13px; color: rgba(255,255,255,0.86); }
    .content { padding: 24px; }
    h2 { margin: 22px 0 10px; font-size: 17px; color: #092541; }
    p { margin: 8px 0; font-size: 14px; }
    ul { margin: 8px 0 8px 20px; padding: 0; font-size: 14px; }
    li { margin: 6px 0; }
    .muted { color: #6b7280; font-size: 13px; }
    .callout { margin-top: 14px; padding: 12px 14px; background: #f3f7fc; border: 1px solid #d9e8fb; border-radius: 10px; }
    .callout strong { color: #092541; }
    .contact { margin-top: 24px; padding: 12px 14px; background: #fff8e6; border: 1px solid #f6d98b; color: #7a5a00; border-radius: 10px; }
    .contact a { color: #092541; font-weight: 700; text-decoration: none; }
    code { background: #f1f5f9; padding: 2px 6px; border-radius: 6px; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="header">
      <h1>AI Email Assistance - Privacy Policy</h1>
      <p>Last updated: Apr 30, 2026</p>
    </div>
    <div class="content">
      <p>This Privacy Policy explains how AI Email Assistance ("we", "us", or "our") collects, uses, and shares information when you use our browser extension, hosted pages, and related services (the "Service").</p>

      <div class="callout">
        <p><strong>Important:</strong> AI Email Assistance helps you draft replies in Gmail. You are responsible for reviewing and sending any message. Please avoid submitting highly sensitive data unless you are authorized to do so.</p>
      </div>

      <h2>INFORMATION WE COLLECT</h2>
      <ul>
        <li><strong>Account identifiers:</strong> if you sign in with Google, we store your Google email address and a Google user identifier to associate usage and plan status.</li>
        <li><strong>Usage data:</strong> we store usage events (for example: generating suggestions, summarizing an email, voice playback/transcription usage) to enforce quotas and provide plan features.</li>
        <li><strong>Content you submit for AI:</strong> when you request AI email details, suggestions, or edits, the text you provide (and relevant email context) is sent to an AI model provider through our server to generate the result.</li>
        <li><strong>Payment metadata (paid plans):</strong> payments are processed by Stripe. We may store Stripe customer and subscription identifiers to keep your plan status in sync.</li>
      </ul>

      <h2>HOW WE USE INFORMATION</h2>
      <ul>
        <li>Provide the Service features (email details, suggestions, translations, voice features).</li>
        <li>Authenticate you and associate your plan status with your account.</li>
        <li>Prevent abuse, enforce quotas, and maintain service reliability.</li>
        <li>Process subscription status updates (upgrade, renewal, cancellation) via Stripe webhooks.</li>
        <li>Improve the Service (debugging, performance, and feature development).</li>
      </ul>

      <h2>HOW WE SHARE INFORMATION</h2>
      <ul>
        <li><strong>AI providers:</strong> when you use AI features, relevant text is sent to the selected AI model provider (for example OpenAI, DeepSeek, or Google Gemini) to generate results.</li>
        <li><strong>Payment provider:</strong> Stripe processes payments and subscription management for paid plans.</li>
        <li><strong>Legal & security:</strong> we may disclose information if required by law, or to protect users, prevent fraud/abuse, or address security incidents.</li>
      </ul>

      <h2>DATA RETENTION</h2>
      <p>We retain account identifiers and usage records for as long as needed to provide the Service, enforce quotas, and meet legal, security, or billing requirements. We may retain limited logs for troubleshooting and abuse prevention.</p>

      <h2>SECURITY</h2>
      <p>We take reasonable measures to protect information in transit and at rest. No system can be 100% secure, so please use the Service responsibly and keep your browser and Google account secure.</p>

      <h2>YOUR CHOICES</h2>
      <ul>
        <li>You can stop using the Service at any time by disabling or uninstalling the extension.</li>
        <li>If you use a paid plan, you can cancel your subscription in Stripe (or via the customer portal if enabled).</li>
        <li>You may contact us to request access, correction, or deletion of your account data, subject to legal and operational constraints.</li>
      </ul>

      <h2>CHANGES TO THIS POLICY</h2>
      <p>We may update this Privacy Policy from time to time. Updates become effective when published on this page.</p>

      <h2>CONTACT US</h2>
      <div class="contact">
        <p>For privacy questions or requests, contact:</p>
        <p><a href="mailto:aiemailassistance@gmail.com">aiemailassistance@gmail.com</a></p>
      </div>

      <p class="muted">For Terms of Use, see <a href="/terms">/terms</a>. For pricing, see <a href="/pricing">/pricing</a>.</p>
    </div>
  </div>
</body>
</html>
    """, 200, {"Content-Type": "text/html; charset=utf-8"}


@app.route("/stripe-success", methods=["GET"])
def stripe_success_page():
    """Professional confirmation page after successful checkout."""
    return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Payment successful</title>
  <style>
    body { margin:0; font-family: Arial, sans-serif; background: linear-gradient(180deg, #eef4fb 0%, #f8fafc 100%); color:#111; }
    .wrap { max-width: 900px; margin: 48px auto; padding: 0 18px; }
    .card { background:#fff; border:1px solid #e5e7eb; border-radius: 18px; box-shadow: 0 18px 48px rgba(9,37,65,0.12); overflow:hidden; }
    .hero { background: linear-gradient(135deg, #092541 0%, #1b4d7a 100%); color:#fff; padding: 26px 24px 22px; }
    .badge { display:inline-flex; align-items:center; gap:8px; background: rgba(255,255,255,0.12); border:1px solid rgba(255,255,255,0.18); color:#fff; border-radius:999px; padding:8px 12px; font-size:12px; font-weight:700; letter-spacing:.02em; }
    .badge-dot { width:10px; height:10px; border-radius:50%; background:#34d399; box-shadow:0 0 0 4px rgba(52,211,153,0.18); }
    h1 { margin:14px 0 8px; font-size: 28px; line-height:1.2; }
    .hero p { margin:0; color: rgba(255,255,255,0.9); font-size: 15px; line-height:1.6; }
    .body { padding: 22px 24px 24px; }
    .intro { margin:0 0 16px; color:#334155; line-height:1.7; font-size:15px; }
    .steps { margin: 0 0 18px; padding: 0; list-style: none; display:grid; gap: 10px; }
    .steps li { display:flex; gap:12px; align-items:flex-start; background:#f8fafc; border:1px solid #e2e8f0; border-radius: 12px; padding: 12px 13px; color:#334155; font-size:14px; line-height:1.6; }
    .step-no { flex:0 0 auto; width:24px; height:24px; border-radius:50%; background:#092541; color:#fff; display:inline-flex; align-items:center; justify-content:center; font-weight:700; font-size:12px; }
    .note { background:#fff8e6; border:1px solid #f6d98b; color:#7a5a00; border-radius:12px; padding:12px 13px; font-size:14px; line-height:1.6; margin-bottom:18px; }
    .btns { display:flex; gap:10px; flex-wrap:wrap; margin-top: 8px; }
    a.btn { display:inline-block; text-decoration:none; padding:11px 15px; border-radius: 12px; font-weight:700; font-size: 13px; transition: transform 120ms ease; }
    a.btn:hover { transform: translateY(-1px); }
    .primary { background:#092541; color:#fff; border:1px solid #092541; }
    .ghost { background:#fff; color:#092541; border:1px solid #cbd5e1; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <div class="hero">
        <div class="badge"><span class="badge-dot"></span> Subscription confirmed</div>
        <h1>Payment successful</h1>
        <p>Your subscription has been created successfully. Your AI Email Assistance plan will be updated automatically in a few seconds.</p>
      </div>
      <div class="body">
        <p class="intro">You can return to Gmail now and continue using AI Email Assistance.</p>
        <ul class="steps">
          <li><span class="step-no">1</span><span>Return to Gmail and try generating again.</span></li>
          <li><span class="step-no">2</span><span>If Gmail is already open, go back to the same tab first.</span></li>
          <li><span class="step-no">3</span><span>If your account still shows <strong>Free</strong>, please refresh the Gmail tab once and try again.</span></li>
        </ul>
        <div class="note"><strong>Important:</strong> If your account still appears on the Free plan after refreshing, wait a few moments and try again.</div>
        <div class="btns">
          <a class="btn primary" href="https://mail.google.com/" target="_blank" rel="noopener">Return to Gmail</a>
          <a class="btn ghost" href="/pricing">Back to pricing</a>
        </div>
      </div>
    </div>
  </div>
</body>
</html>
    """, 200, {"Content-Type": "text/html; charset=utf-8"}


@app.route("/stripe-cancel", methods=["GET"])
def stripe_cancel_page():
    """Landing page after cancelled checkout."""
    return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Checkout cancelled</title>
  <style>
    body { margin:0; font-family: Arial, sans-serif; background:#f7f8fb; color:#111; }
    .wrap { max-width: 820px; margin: 36px auto; padding: 0 16px; }
    .card { background:#fff; border:1px solid #e5e7eb; border-radius: 14px; box-shadow: 0 10px 28px rgba(0,0,0,0.08); padding: 18px 18px 16px; }
    h1 { margin:0 0 8px; color:#092541; font-size: 22px; }
    p { margin:0 0 12px; color:#444; line-height:1.6; font-size: 14px; }
    a.btn { display:inline-block; text-decoration:none; padding:10px 14px; border-radius: 10px; font-weight:700; font-size: 13px; background:#092541; color:#fff; border:1px solid #092541; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>Checkout cancelled</h1>
      <p>No payment was made. You can try again anytime.</p>
      <a class="btn" href="/pricing">Back to pricing</a>
    </div>
  </div>
</body>
</html>
    """, 200, {"Content-Type": "text/html; charset=utf-8"}


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)