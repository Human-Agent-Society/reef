"""Judge-path reverse proxy for the SkillClaw reproduction (v2, streaming).

The official code sizes judge/evolver max_tokens for a judge whose whole
completion is the answer. The served GLM-5.3-Flash is a reasoning model:
reasoning tokens count against max_tokens, so small official budgets (600,
1200, 2048, 4096, 8192) return empty content deterministically. Graders,
the ported evolver, and openclaw are official code at a pin, so the
accommodation lives here: raise small completion budgets and forward
everything else byte for byte. v2 added SSE passthrough so the agent day
path (streamed through the Reef service) can also route here. v3 covers
both budget field names.

Declared deviations, applied uniformly to both runs:
- max_tokens below 32768 is raised to 32768 (graders 600-2048, night judge
  1200, night decide/create/merge 8192 whose measured need reaches ~15.2k
  on the largest skill, and agent-authored small calls).
- max_completion_tokens below 16384 is raised to 16384 (openclaw image and
  pdf tools hard-cap at 4096, emptied by reasoning). The floor is chosen so
  openclaw's main-path budget of 32000 is never touched.
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM = os.environ.get("REEF_SC_UPSTREAM_DIRECT", "http://127.0.0.1:30001")
LISTEN = ("0.0.0.0", int(sys.argv[1]) if len(sys.argv) > 1 else 30011)
MIN_MAX_TOKENS = 32768
MIN_MAX_COMPLETION_TOKENS = 16384
CHUNK = 8192
# Passive tap: night-pipeline calls (decide/create/merge/judge) are not
# recorded by the reef store, so their raw replies are otherwise lost.
# Pure observability; request and response bytes are forwarded unchanged.
TAP_MARKERS = (b"Existing skill names in the library", b"session-level evaluator")
TAP_FILE = os.environ.get("REEF_SC_NIGHT_TAP", "night-tap.jsonl")
# Declared accommodation (2026-08-29): GLM-5.3-Flash drops one closing brace
# on long improve_skill JSON replies (finish=stop, depth short by 1), which
# the official zero-tolerance parser turns into a silent skip - the improve
# channel measured 0/3 dead across nights 1-2 while create ran 2/2. The
# official experiment ran with a working improve channel, so constrained
# decoding restores that function. Applied only to the evolver's
# decide/create/merge stages (system prompt prefix below); the judge and
# ALL day-path traffic are untouched. Verified on the exact tapped failing
# requests: 6/6 parse OK with same-scale content.
NIGHT_STAGE_PREFIX = "You are a skill engineer for SkillClaw"


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def _relay(self, resp, status):
        """Stream the upstream body through as it arrives (close-delimited)."""
        self.send_response(status)
        ctype = resp.headers.get("Content-Type", "application/json")
        self.send_header("Content-Type", ctype)
        self.send_header("Connection", "close")
        self.end_headers()
        while True:
            chunk = resp.read(CHUNK)
            if not chunk:
                break
            self.wfile.write(chunk)
            self.wfile.flush()

    def _forward(self, body=None, tap=False):
        url = UPSTREAM + self.path
        headers = {k: v for k, v in self.headers.items() if k.lower() not in ("host", "content-length", "connection")}
        req = urllib.request.Request(url, data=body, headers=headers, method=self.command)
        try:
            resp = urllib.request.urlopen(req, timeout=580)
        except urllib.error.HTTPError as err:
            payload = err.read()
            self.send_response(err.code)
            self.send_header("Content-Type", err.headers.get("Content-Type", "application/json"))
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(payload)
            return
        except Exception as err:
            payload = json.dumps({"error": {"message": str(err)}}).encode()
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(payload)
            return
        try:
            if tap:
                payload = resp.read()
                try:
                    import json as _json

                    with open(TAP_FILE, "a") as tf:
                        tf.write(
                            _json.dumps(
                                {
                                    "ts": time.time(),
                                    "status": resp.status,
                                    "request": (body or b"").decode("utf-8", "replace"),
                                    "response": payload.decode("utf-8", "replace"),
                                }
                            )
                            + "\n"
                        )
                except Exception:
                    pass  # the tap must never break the forward path
                self.send_response(resp.status)
                self.send_header("Content-Type", resp.headers.get("Content-Type", "application/json"))
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(payload)
            else:
                self._relay(resp, resp.status)
        except (BrokenPipeError, ConnectionResetError):
            pass  # client went away mid-stream; upstream request already ran
        finally:
            resp.close()

    def do_GET(self):
        self._forward()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length else b""
        if self.path.endswith("/chat/completions"):
            try:
                payload = json.loads(body)
                changed = []
                for field, floor in (
                    ("max_tokens", MIN_MAX_TOKENS),
                    ("max_completion_tokens", MIN_MAX_COMPLETION_TOKENS),
                ):
                    asked = payload.get(field)
                    if isinstance(asked, int) and asked < floor:
                        payload[field] = floor
                        changed.append(f"{field} {asked} -> {floor}")
                if changed:
                    body = json.dumps(payload).encode()
                    print("bumped " + "; ".join(changed), flush=True)
                msgs = payload.get("messages") or []
                first = msgs[0] if msgs else {}
                if (
                    isinstance(first, dict)
                    and str(first.get("content", "")).startswith(NIGHT_STAGE_PREFIX)
                    and "response_format" not in payload
                ):
                    payload["response_format"] = {"type": "json_object"}
                    body = json.dumps(payload).encode()
                    print("constrained night-stage call to json_object", flush=True)
            except (ValueError, TypeError):
                pass
        tap = self.path.endswith("/chat/completions") and any(m in body for m in TAP_MARKERS)
        self._forward(body, tap=tap)


if __name__ == "__main__":
    print(f"judge proxy v2 on {LISTEN} -> {UPSTREAM}, min max_tokens {MIN_MAX_TOKENS}", flush=True)
    ThreadingHTTPServer(LISTEN, Handler).serve_forever()
