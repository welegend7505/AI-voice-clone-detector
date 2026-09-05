"""
SQLite persistence for speakers, calls, and per-chunk risk history.

Replaces the in-memory dicts in api/routes.py (P1, 2026-09-06) so that a
server restart mid-demo no longer loses enrollments or call state, and
risk history survives for later inspection.

Design constraints (hackathon, 2am debuggability):
  - stdlib sqlite3 only, no ORM, one .db file next to main.py
    (override with env SIH_DB_PATH, e.g. ":memory:" for tests)
  - enrollment audio is stored as a BLOB (a ~20 s 16 kHz wav is ~600 KB,
    trivial for sqlite) and materialized to enrolled_speakers/<id>.wav
    on demand, because the ML models score from file paths
  - one module-level connection + a lock: FastAPI handlers are async but
    sqlite calls here are sub-millisecond, so blocking briefly is fine
"""

import json
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone

DB_PATH = os.environ.get("SIH_DB_PATH",
                         os.path.join(os.path.dirname(os.path.abspath(__file__)), "sih.db"))
_ENROLL_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "enrolled_speakers")

_conn: sqlite3.Connection = None
_lock = threading.Lock()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS speakers(
    speaker_id  TEXT PRIMARY KEY,
    wav         BLOB NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS calls(
    call_id     TEXT PRIMARY KEY,
    speaker_id  TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS risk_history(
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    call_id           TEXT NOT NULL,
    ts                TEXT NOT NULL,
    risk_score        REAL,
    spoof_risk        REAL,
    speaker_mismatch  REAL,
    prosody_anomaly   REAL,
    alert             INTEGER,
    flags             TEXT,
    aasist_logit      REAL,
    w2v2_logit        REAL
);
CREATE INDEX IF NOT EXISTS idx_history_call ON risk_history(call_id, id);
"""


def init_db() -> None:
    """Create tables if missing. Safe to call repeatedly."""
    global _conn
    with _lock:
        if _conn is None:
            _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
            _conn.row_factory = sqlite3.Row
        _conn.executescript(_SCHEMA)
        _conn.commit()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------- speakers ----------------

def save_speaker(wav_bytes: bytes) -> str:
    """Store a new enrollment; returns the generated speaker_id."""
    init_db()
    speaker_id = str(uuid.uuid4())
    with _lock:
        _conn.execute("INSERT INTO speakers VALUES (?, ?, ?)",
                      (speaker_id, wav_bytes, _now()))
        _conn.commit()
    return speaker_id


def speaker_exists(speaker_id: str) -> bool:
    init_db()
    with _lock:
        row = _conn.execute("SELECT 1 FROM speakers WHERE speaker_id = ?",
                            (speaker_id,)).fetchone()
    return row is not None


def get_speaker_wav_path(speaker_id: str):
    """
    Materialize the stored enrollment wav to a stable cache path and return
    it (the models score from file paths). Returns None if unknown speaker.
    The cache file is reused across calls; rewrite if missing/corrupt.
    """
    init_db()
    with _lock:
        row = _conn.execute("SELECT wav FROM speakers WHERE speaker_id = ?",
                            (speaker_id,)).fetchone()
    if row is None:
        return None
    os.makedirs(_ENROLL_CACHE_DIR, exist_ok=True)
    path = os.path.join(_ENROLL_CACHE_DIR, f"{speaker_id}.wav")
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "wb") as f:
            f.write(row["wav"])
    return path


# ---------------- calls ----------------

def create_call(speaker_id: str) -> str:
    init_db()
    call_id = str(uuid.uuid4())
    with _lock:
        _conn.execute("INSERT INTO calls VALUES (?, ?, ?)",
                      (call_id, speaker_id, _now()))
        _conn.commit()
    return call_id


def get_call(call_id: str):
    """Returns {"call_id","speaker_id","created_at"} or None."""
    init_db()
    with _lock:
        row = _conn.execute("SELECT * FROM calls WHERE call_id = ?",
                            (call_id,)).fetchone()
    return dict(row) if row else None


# ---------------- risk history ----------------

def append_history(call_id: str, result: dict) -> None:
    """Store one scored-chunk result (the WS response dict)."""
    init_db()
    with _lock:
        _conn.execute(
            """INSERT INTO risk_history
               (call_id, ts, risk_score, spoof_risk, speaker_mismatch,
                prosody_anomaly, alert, flags, aasist_logit, w2v2_logit)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (call_id, _now(),
             result.get("risk_score"), result.get("spoof_risk"),
             result.get("speaker_mismatch"), result.get("prosody_anomaly"),
             int(bool(result.get("alert"))),
             json.dumps(result.get("flags", [])),
             result.get("aasist_logit_raw"), result.get("w2v2_logit_raw")))
        _conn.commit()


def latest_risk(call_id: str):
    """Rebuild the newest result dict for a call, or None if no history."""
    init_db()
    with _lock:
        row = _conn.execute(
            "SELECT * FROM risk_history WHERE call_id = ? ORDER BY id DESC LIMIT 1",
            (call_id,)).fetchone()
    if row is None:
        return None
    return latest_risk_from_row(row)


def history(call_id: str, limit: int = 100):
    """Newest-first list of result dicts (same shape as latest_risk)."""
    init_db()
    with _lock:
        rows = _conn.execute(
            "SELECT * FROM risk_history WHERE call_id = ? ORDER BY id DESC LIMIT ?",
            (call_id, limit)).fetchall()
    return [latest_risk_from_row(r) for r in rows]


def latest_risk_from_row(row) -> dict:
    return {
        "call_id": row["call_id"],
        "risk_score": row["risk_score"],
        "spoof_risk": row["spoof_risk"],
        "speaker_mismatch": row["speaker_mismatch"],
        "prosody_anomaly": row["prosody_anomaly"],
        "flags": json.loads(row["flags"] or "[]"),
        "alert": bool(row["alert"]),
        "aasist_logit_raw": row["aasist_logit"],
        "w2v2_logit_raw": row["w2v2_logit"],
    }
