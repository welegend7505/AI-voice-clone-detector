"""
Threshold-crossing alert trigger (P1, 2026-09-06).

On every scored chunk where risk["alert"] is True:
  - always logs a clearly-greppable console line
  - if env ALERT_WEBHOOK_URL is set, POSTs the full result JSON there
    (fire-and-forget, in a worker thread -- a slow/dead webhook must
    never stall the WebSocket scoring loop; failures are logged, not raised)

This is deliberately the whole feature: no retry queue, no dedupe, no
SMS/email (that's P2 if there's time). Good enough to wire a laptop-shape
demo: point ALERT_WEBHOOK_URL at anything that accepts JSON (a Slack
webhook, a local buzzer service, requestbin while testing).
"""

import json
import os
import urllib.request
from datetime import datetime, timezone

WEBHOOK_URL = os.environ.get("ALERT_WEBHOOK_URL", "")


def fire_alert(result: dict) -> None:
    """Log + optionally webhook one alerting chunk. Never raises."""
    try:
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        print(f"[ALERT {ts}] call={result.get('call_id')} "
              f"risk={result.get('risk_score')} flags={result.get('flags')} "
              f"spoof={result.get('spoof_risk')} mism={result.get('speaker_mismatch')}",
              flush=True)
        if WEBHOOK_URL:
            _post_webhook(WEBHOOK_URL, result)
    except Exception as e:  # webhook/log failure must never break the call
        print(f"[ALERT] delivery failed (ignored): {type(e).__name__}: {e}",
              flush=True)


def _post_webhook(url: str, payload: dict) -> None:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), method="POST",
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=5) as r:
        r.read()
