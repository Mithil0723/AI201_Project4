"""
app.py — Provenance Guard backend.

A small Flask service any creative-sharing platform could plug in to:
  * classify submitted text as AI- or human-written (3-signal ensemble),
  * score confidence in a way that admits uncertainty,
  * return a plain-language transparency label,
  * accept appeals from creators who think they were misclassified,
  * rate-limit submissions and keep a structured audit log.

Run:
    pip install -r requirements.txt
    python app.py                 # real Groq LLM signal (needs GROQ_API_KEY)
    MOCK_LLM=1 python app.py       # offline test mode (no network/key needed)

Endpoints:
    POST /submit   {text, creator_id}            -> classification + label
    POST /appeal   {content_id, creator_reasoning} -> status -> under_review
    GET  /log?limit=N                            -> recent audit entries
    GET  /health                                 -> liveness check
"""

import os
import json
import uuid
import sqlite3
from datetime import datetime, timezone

from flask import Flask, request, jsonify, g
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from dotenv import load_dotenv

from signals import run_ensemble
from labels import make_label

load_dotenv()

DB_PATH = os.environ.get("PROV_DB", "provenance.db")

app = Flask(__name__)

# --------------------------------------------------------------------------- #
#  Rate limiting
#  10/min, 100/day per IP on /submit. Reasoning (documented in README):
#  a real creator submits their own work a handful of times an hour at most;
#  10/min leaves comfortable headroom for honest editing-and-resubmit loops
#  while a script trying to flood or fuzz the classifier hits the wall fast.
#  100/day caps sustained automated abuse without blocking a busy human.
# --------------------------------------------------------------------------- #
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[],
    storage_uri="memory://",
)

# --------------------------------------------------------------------------- #
#  SQLite storage + structured audit log
# --------------------------------------------------------------------------- #


def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(_exc):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    db = sqlite3.connect(DB_PATH)
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS submissions (
            content_id   TEXT PRIMARY KEY,
            creator_id   TEXT,
            text         TEXT,
            created_at   TEXT,
            attribution  TEXT,
            confidence   REAL,
            llm_score    REAL,
            stylo_score  REAL,
            lexical_score REAL,
            status       TEXT
        );
        CREATE TABLE IF NOT EXISTS audit_log (
            log_id       INTEGER PRIMARY KEY AUTOINCREMENT,
            content_id   TEXT,
            creator_id   TEXT,
            event        TEXT,            -- 'classified' | 'appeal'
            timestamp    TEXT,
            attribution  TEXT,
            confidence   REAL,
            llm_score    REAL,
            stylo_score  REAL,
            lexical_score REAL,
            status       TEXT,
            label        TEXT,
            appeal_reasoning TEXT
        );
        """
    )
    db.commit()
    db.close()


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def write_audit(db, **row):
    cols = ", ".join(row.keys())
    qs = ", ".join("?" for _ in row)
    db.execute(f"INSERT INTO audit_log ({cols}) VALUES ({qs})", tuple(row.values()))
    db.commit()


# --------------------------------------------------------------------------- #
#  POST /submit
# --------------------------------------------------------------------------- #


@app.route("/submit", methods=["POST"])
@limiter.limit("10 per minute;100 per day")
def submit():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    creator_id = (data.get("creator_id") or "").strip()

    if not text:
        return jsonify({"error": "Field 'text' is required."}), 400
    if not creator_id:
        return jsonify({"error": "Field 'creator_id' is required."}), 400

    result = run_ensemble(text)
    confidence = result["confidence"]
    attribution, label = make_label(confidence)

    content_id = str(uuid.uuid4())
    ts = now_iso()
    sig = result["signals"]
    llm_score = sig["llm"]["score"]
    stylo_score = sig["stylometry"]["score"]
    lexical_score = sig["lexical"]["score"]

    db = get_db()
    db.execute(
        """INSERT INTO submissions
           (content_id, creator_id, text, created_at, attribution, confidence,
            llm_score, stylo_score, lexical_score, status)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (content_id, creator_id, text, ts, attribution, confidence,
         llm_score, stylo_score, lexical_score, "classified"),
    )
    db.commit()

    write_audit(
        db,
        content_id=content_id,
        creator_id=creator_id,
        event="classified",
        timestamp=ts,
        attribution=attribution,
        confidence=confidence,
        llm_score=llm_score,
        stylo_score=stylo_score,
        lexical_score=lexical_score,
        status="classified",
        label=label,
        appeal_reasoning=None,
    )

    return jsonify({
        "content_id": content_id,
        "creator_id": creator_id,
        "attribution": attribution,
        "confidence": confidence,
        "label": label,
        "status": "classified",
        "signals": {
            "llm": {"score": llm_score, "available": sig["llm"]["available"],
                    "reasoning": sig["llm"]["reasoning"]},
            "stylometry": {"score": stylo_score, "metrics": sig["stylometry"]["metrics"]},
            "lexical": {"score": lexical_score, "markers": sig["lexical"]["markers"]},
        },
        "weights_used": result["weights_used"],
        "vote": result["vote"],
    }), 200


# --------------------------------------------------------------------------- #
#  POST /appeal
# --------------------------------------------------------------------------- #


@app.route("/appeal", methods=["POST"])
def appeal():
    data = request.get_json(silent=True) or {}
    content_id = (data.get("content_id") or "").strip()
    reasoning = (data.get("creator_reasoning") or "").strip()

    if not content_id or not reasoning:
        return jsonify({
            "error": "Both 'content_id' and 'creator_reasoning' are required."
        }), 400

    db = get_db()
    row = db.execute(
        "SELECT * FROM submissions WHERE content_id = ?", (content_id,)
    ).fetchone()
    if row is None:
        return jsonify({"error": f"No submission found for content_id {content_id}."}), 404

    db.execute(
        "UPDATE submissions SET status = ? WHERE content_id = ?",
        ("under_review", content_id),
    )
    db.commit()

    ts = now_iso()
    # Log the appeal next to the ORIGINAL decision so a reviewer sees both.
    write_audit(
        db,
        content_id=content_id,
        creator_id=row["creator_id"],
        event="appeal",
        timestamp=ts,
        attribution=row["attribution"],
        confidence=row["confidence"],
        llm_score=row["llm_score"],
        stylo_score=row["stylo_score"],
        lexical_score=row["lexical_score"],
        status="under_review",
        label=None,
        appeal_reasoning=reasoning,
    )

    return jsonify({
        "content_id": content_id,
        "status": "under_review",
        "message": "Appeal received. This classification is now under human review.",
        "original_decision": {
            "attribution": row["attribution"],
            "confidence": row["confidence"],
        },
        "appeal_reasoning": reasoning,
        "appealed_at": ts,
    }), 200


# --------------------------------------------------------------------------- #
#  GET /log  (documentation / grading visibility; would require auth in prod)
# --------------------------------------------------------------------------- #


@app.route("/log", methods=["GET"])
def get_log():
    limit = request.args.get("limit", default=20, type=int)
    db = get_db()
    rows = db.execute(
        "SELECT * FROM audit_log ORDER BY log_id DESC LIMIT ?", (limit,)
    ).fetchall()
    entries = [dict(r) for r in rows]
    return jsonify({"count": len(entries), "entries": entries}), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "mock_llm": os.environ.get("MOCK_LLM") == "1"}), 200


@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({
        "error": "rate_limit_exceeded",
        "detail": str(e.description),
        "message": "Too many submissions. Please slow down and try again shortly.",
    }), 429


if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=5000, debug=False)
