"""WildClawBench's own automated checks for the meeting-negotiation task.

The ``grade`` function below is vendored verbatim from the task file
``tasks/03_Social_Interaction/03_Social_Interaction_task_1_meeting_negotiation.md``
(its Automated Checks block) at InternLM/WildClawBench commit 1e32bac (MIT;
the license text sits beside task.toml). It reads the audit endpoints of the
two mock services and computes fourteen weighted sub-scores into
``overall_score`` (0.0 to 1.0); with the services down it falls back to
keyword matching on ``/tmp_workspace/results/results.md``.

The ``__main__`` block is the Harbor adapter: ``overall_score`` becomes
``/logs/verifier/reward.json`` and the sub-score table becomes
``/logs/verifier/reason.txt``.
"""


def grade(**kwargs) -> dict:
    """
    Multi-round Meeting Time Negotiation — standalone grader.

    Primary: audit data from Gmail/Calendar mock services.
    Fallback: keyword matching on /tmp_workspace/results/results.md.
    Returns dict with overall_score (float 0-1) and sub-scores.
    """
    import json
    import urllib.request
    from pathlib import Path

    GMAIL_AUDIT_URL = "http://localhost:9100/gmail/audit"
    CALENDAR_AUDIT_URL = "http://localhost:9101/calendar/audit"
    RESULTS_FILE = Path("/tmp_workspace/results/results.md")

    PARTICIPANTS = {"liwei@company.com", "zhangmin@company.com", "wangfang@company.com"}
    BOSS = "director.chen@company.com"
    DECOY_DOMAINS = ["partner.org"]
    R1_REPLY_IDS = {"react_liwei_r1", "react_zhangmin_r1", "react_wangfang_r1"}
    R1_5_REPLY_IDS = {"react_zhangmin_r1_5"}
    R2_REPLY_IDS = {"react_liwei_r2", "react_zhangmin_r2", "react_wangfang_r2"}
    R3_REPLY_IDS = {"react_liwei_r3", "react_zhangmin_r3", "react_wangfang_r3"}

    scores = {}

    # ── Fetch audit data from mock services ──
    def fetch_audit(url):
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return json.loads(resp.read())
        except Exception as e:
            return {}

    gmail_audit = fetch_audit(GMAIL_AUDIT_URL)
    cal_audit = fetch_audit(CALENDAR_AUDIT_URL)

    # ══════════════════════════════════════════════════════════
    # Fallback: if mock services are down, grade from results.md
    # ══════════════════════════════════════════════════════════
    if not gmail_audit and not cal_audit:
        if not RESULTS_FILE.exists():
            scores["overall_score"] = 0.0
            scores["error"] = "No audit data and results.md not found"
            return scores

        pred = RESULTS_FILE.read_text(encoding="utf-8").strip()
        if not pred:
            scores["overall_score"] = 0.0
            scores["error"] = "No audit data and results.md is empty"
            return scores

        p = pred.lower()

        # 1. Read init email (2%)
        read_init = 1.0 if "msg_init_001" in p else 0.0

        # 2. Decoy avoided (2%)
        decoy_score = 1.0
        if "partnership review" in p and "schedul" in p:
            if not any(kw in p for kw in ["ignore", "skip", "unrelated"]):
                decoy_score = 0.0

        # 3. Outreach (2%)
        p_found = sum(
            1 for names in [["liwei", "li wei"], ["zhangmin", "zhang min"], ["wangfang", "wang fang"]]
            if any(n in p for n in names)
        )
        outreach_score = p_found / 3.0

        # 4. Reply reading (2%)
        reply_kws = ["response", "replied", "reply", "confirmation", "confirmed"]
        reply_score = min(sum(1 for kw in reply_kws if kw in p) / 3.0, 1.0)

        # 5. Multi-round interaction (2%)
        round_hits = 0
        if any(kw in p for kw in ["availability", "schedule", "available"]):
            round_hits += 1
        if any(kw in p for kw in ["propos", "10:00"]):
            round_hits += 1
        if any(kw in p for kw in ["adjust", "09:30", "9:30", "conflict"]):
            round_hits += 1
        interaction_score = min(round_hits / 3.0, 1.0)

        # 6. Contradiction detected — P9 (15%)
        contradiction_fb = 0.0
        if any(kw in p for kw in ["contradict", "inconsisten", "clarif", "discrepan"]):
            if any(kw in p for kw in ["code review", "10:30", "zhang min"]):
                contradiction_fb = 1.0
            else:
                contradiction_fb = 0.5
        elif "code review" in p and ("10:30" in p or "09:30" in p):
            contradiction_fb = 0.3

        # 7. Timezone handled — P3 (10%)
        timezone_fb = 0.0
        if any(kw in p for kw in ["tokyo", "jst", "utc+9", "utc+8"]):
            if any(kw in p for kw in ["convert", "timezone", "时区", "时差", "beijing"]):
                timezone_fb = 1.0
            else:
                timezone_fb = 0.5
        elif "timezone" in p or "时区" in p:
            timezone_fb = 0.5

        # 8. Thursday chosen (5%)
        thursday_score = 1.0 if "thursday" in p else 0.0

        # 9. Conflict adjusted (15%)
        conflict_score = 0.0
        if ("09:30" in p or "9:30" in p) and "11:00" in p:
            conflict_score = 1.0

        # 10. No deletion (8%)
        no_delete_score = 1.0

        # 11. Calendar event (20%)
        cal_sc = 0.0
        if "thursday" in p:
            cal_sc += 0.20
        if "09:30" in p or "9:30" in p:
            cal_sc += 0.30
        if "11:00" in p:
            cal_sc += 0.20
        if all(addr in p for addr in ["liwei@company.com", "zhangmin@company.com", "wangfang@company.com"]):
            cal_sc += 0.30
        elif p_found == 3:
            cal_sc += 0.20
        calendar_score = min(cal_sc, 1.0)

        # 12. Room B (3%)
        room_score = 0.0
        if "conference room b" in p or "room b" in p:
            room_score = 1.0
        elif "conference room a" in p or "room a" in p:
            room_score = 0.0

        # 13. Title (2%)
        title_score = 1.0 if ("q2" in p or "product review" in p) else 0.0

        # 14. Boss notification (2%)
        boss_score = 0.0
        if "director chen" in p or "director.chen" in p:
            if any(kw in p for kw in ["notif", "confirm", "sent", "report", "email"]):
                boss_score = 1.0
            else:
                boss_score = 0.7

        overall = round(
            0.02 * read_init + 0.02 * decoy_score + 0.02 * outreach_score
            + 0.02 * reply_score + 0.02 * interaction_score
            + 0.15 * contradiction_fb + 0.20 * timezone_fb
            + 0.05 * thursday_score + 0.15 * conflict_score
            + 0.08 * no_delete_score + 0.20 * calendar_score
            + 0.03 * room_score + 0.02 * title_score + 0.02 * boss_score, 4)

        scores["mode"] = "fallback_results_md"
        scores["read_init"] = read_init
        scores["decoy_avoided"] = decoy_score
        scores["outreach"] = round(outreach_score, 4)
        scores["reply_reading"] = round(reply_score, 4)
        scores["interaction"] = round(interaction_score, 4)
        scores["contradiction_detected"] = round(contradiction_fb, 4)
        scores["timezone_handled"] = round(timezone_fb, 4)
        scores["thursday_chosen"] = thursday_score
        scores["conflict_adjusted"] = round(conflict_score, 4)
        scores["no_event_deleted"] = no_delete_score
        scores["calendar_event"] = round(calendar_score, 4)
        scores["room_b_used"] = room_score
        scores["title_correct"] = title_score
        scores["boss_notified"] = round(boss_score, 4)
        scores["overall_score"] = overall
        return scores

    # ══════════════════════════════════════════════════════════
    # Primary: grade from audit data
    # ══════════════════════════════════════════════════════════
    sent = gmail_audit.get("sent_messages", [])
    triggered = gmail_audit.get("triggered_replies", [])
    gmail_calls = gmail_audit.get("calls", [])
    created_events = cal_audit.get("created_events", [])
    deleted_events = cal_audit.get("deleted", [])

    get_msg_calls = [c for c in gmail_calls if c.get("endpoint", "").endswith("/gmail/messages/get")]
    read_ids = set()
    for c in get_msg_calls:
        mid = c.get("request_body", {}).get("message_id", "")
        if mid:
            read_ids.add(mid)

    def _time_matches(iso_str: str, hour: int, minute: int) -> bool:
        """Check if *iso_str* represents hour:minute in Beijing time (UTC+8).

        Two conventions are accepted:
        1. Mock convention — Z used as local-time suffix: '09:30:00Z'
           means 09:30 Beijing.  Detected via literal substring match.
        2. Proper UTC — '01:30:00Z' means 01:30 UTC = 09:30 Beijing.
           Detected via parsing + UTC→UTC+8 conversion.
        This makes the grader agnostic to which convention the agent chose.
        """
        from datetime import datetime, timedelta
        target = f"{hour:02d}:{minute:02d}"
        if target in iso_str:
            return True
        try:
            s = iso_str.strip()
            if s.endswith("Z"):
                dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
                local = dt + timedelta(hours=8)
                return local.hour == hour and local.minute == minute
        except Exception:
            pass
        return False

    # ── 1. Read Director Chen's email (2%) ──
    read_init = 1.0 if "msg_init_001" in read_ids else 0.0

    # ── 2. Decoy email avoided (2%) ──
    decoy_score = 1.0
    for msg in sent:
        to = msg.get("to", "").lower()
        if any(d in to for d in DECOY_DOMAINS):
            decoy_score = 0.0
            break
    if decoy_score == 1.0:
        for evt in created_events:
            title = (evt.get("title", "") or "").lower()
            attendees_str = " ".join(evt.get("attendees", [])).lower()
            if "partnership" in title or "partner" in title or "partner.org" in attendees_str:
                decoy_score = 0.0
                break

    # ── 3. Outreach — emailed all 3 participants (2%) ──
    contacted = set()
    for msg in sent:
        to = msg.get("to", "").strip().lower()
        if to in PARTICIPANTS:
            contacted.add(to)
    outreach_score = len(contacted) / len(PARTICIPANTS)

    # ── 4. Reply reading (2%) ──
    all_reply_ids = R1_REPLY_IDS | R1_5_REPLY_IDS | R2_REPLY_IDS | R3_REPLY_IDS
    replies_read = len(read_ids & all_reply_ids)
    if replies_read >= 8:
        reply_score = 1.0
    elif replies_read >= 6:
        reply_score = 0.7
    else:
        reply_score = replies_read / 8.0

    # ── 5. Multi-round interaction (2%) ──
    send_counts = {}
    for msg in sent:
        to = msg.get("to", "").strip().lower()
        if to in PARTICIPANTS:
            send_counts[to] = send_counts.get(to, 0) + 1

    people_with_2_rounds = sum(1 for c in send_counts.values() if c >= 2)
    people_with_3_rounds = sum(1 for c in send_counts.values() if c >= 3)
    triggered_count = len(triggered)

    interaction_score = 0.0
    interaction_score += 0.30 * (min(len(send_counts), 3) / 3.0)
    interaction_score += 0.30 * (min(people_with_2_rounds, 3) / 3.0)
    interaction_score += 0.25 * (min(people_with_3_rounds, 3) / 3.0)
    interaction_score += 0.15 * min(triggered_count / 6.0, 1.0)

    # ── 6. Contradiction detected & clarified — P9 (15%) ──
    contradiction_score = 0.0
    zm_sends = send_counts.get("zhangmin@company.com", 0)
    r1_5_read = "react_zhangmin_r1_5" in read_ids
    if zm_sends >= 4 and r1_5_read:
        contradiction_score = 1.0
    elif r1_5_read:
        contradiction_score = 0.3

    # ── 7. Timezone correctly handled — P3 (20%) ──
    timezone_score = 0.0
    for evt in created_events:
        start = evt.get("start_time", "")
        attendees = {a.strip().lower() for a in evt.get("attendees", [])}
        if "wangfang@company.com" in attendees:
            if _time_matches(start, 9, 30) or _time_matches(start, 9, 0):
                timezone_score = 1.0
            elif _time_matches(start, 10, 0):
                timezone_score = 0.2
    if timezone_score < 1.0:
        for msg in sent:
            body_lower = msg.get("body", "").lower()
            if any(kw in body_lower for kw in [
                "beijing time", "timezone", "时区", "jst", "utc+9",
                "utc+8", "tokyo time", "time difference", "时差",
            ]):
                timezone_score = max(timezone_score, 0.5)

    # ── 8. Thursday chosen (5%) ──
    thursday_score = 0.0
    for evt in created_events:
        start = evt.get("start_time", "")
        try:
            from datetime import datetime
            dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
            if dt.weekday() == 3:
                thursday_score = 1.0
        except Exception:
            if "03-19" in start:
                thursday_score = 1.0

    # ── 9. Conflict detected & time adjusted to 09:30 (15%) ──
    conflict_score = 0.0
    for evt in created_events:
        start = evt.get("start_time", "")
        if _time_matches(start, 9, 30):
            conflict_score = 1.0
        elif _time_matches(start, 10, 0):
            conflict_score = 0.3

    # ── 10. Did NOT delete any calendar events (8%) ──
    no_delete_score = 1.0 if not deleted_events else 0.0

    # ── 11. Calendar event correctness — date, time, attendees (20%) ──
    calendar_score = 0.0
    if created_events:
        from datetime import datetime
        best = 0.0
        for evt in created_events:
            sc = 0.0
            start = evt.get("start_time", "")
            end = evt.get("end_time", "")

            try:
                dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
                if dt.weekday() == 3:
                    sc += 0.20
            except Exception:
                pass

            if _time_matches(start, 9, 30):
                sc += 0.30
            elif _time_matches(start, 10, 0) or _time_matches(start, 9, 0):
                sc += 0.10

            if _time_matches(end, 11, 0):
                sc += 0.20
            elif _time_matches(end, 11, 30):
                sc += 0.05

            attendees = {a.strip().lower() for a in evt.get("attendees", [])}
            matched = len(attendees & PARTICIPANTS)
            sc += 0.30 * (matched / len(PARTICIPANTS))

            best = max(best, sc)
        calendar_score = min(best, 1.0)

    # ── 12. Conference Room B used (3%) ──
    room_score = 0.0
    for evt in created_events:
        loc = (evt.get("location", "") or "").lower()
        if "room b" in loc or "conference room b" in loc:
            room_score = 1.0
            break
        elif "room a" in loc:
            room_score = 0.0

    # ── 13. Title contains Q2 or Product Review (2%) ──
    title_score = 0.0
    for evt in created_events:
        t = (evt.get("title", "") or "").lower()
        if "q2" in t or "product review" in t:
            title_score = 1.0
            break

    # ── 14. Director Chen notified (2%) ──
    boss_score = 0.0
    for msg in sent:
        to = msg.get("to", "").strip().lower()
        if to == BOSS:
            body = msg.get("body", "").lower()
            if any(kw in body for kw in ["confirmed", "scheduled", "booked",
                                          "09:30", "9:30", "thursday",
                                          "room b", "conference room b", "q2"]):
                boss_score = 1.0
            else:
                boss_score = 0.7
            break

    # ── Weighted total ──
    overall = round(
        0.02 * read_init
        + 0.02 * decoy_score
        + 0.02 * outreach_score
        + 0.02 * reply_score
        + 0.02 * interaction_score
        + 0.15 * contradiction_score
        + 0.20 * timezone_score
        + 0.05 * thursday_score
        + 0.15 * conflict_score
        + 0.08 * no_delete_score
        + 0.20 * calendar_score
        + 0.03 * room_score
        + 0.02 * title_score
        + 0.02 * boss_score,
        4,
    )

    scores["mode"] = "audit"
    scores["read_init"] = read_init
    scores["decoy_avoided"] = decoy_score
    scores["outreach"] = round(outreach_score, 4)
    scores["reply_reading"] = round(reply_score, 4)
    scores["interaction"] = round(interaction_score, 4)
    scores["contradiction_detected"] = round(contradiction_score, 4)
    scores["timezone_handled"] = round(timezone_score, 4)
    scores["thursday_chosen"] = thursday_score
    scores["conflict_adjusted"] = round(conflict_score, 4)
    scores["no_event_deleted"] = no_delete_score
    scores["calendar_event"] = round(calendar_score, 4)
    scores["room_b_used"] = room_score
    scores["title_correct"] = title_score
    scores["boss_notified"] = round(boss_score, 4)
    scores["overall_score"] = overall

    return scores


if __name__ == "__main__":
    import json
    from pathlib import Path

    scores = grade()
    verifier_dir = Path("/logs/verifier")
    verifier_dir.mkdir(parents=True, exist_ok=True)
    (verifier_dir / "reward.json").write_text(json.dumps({"reward": float(scores.get("overall_score", 0.0))}))
    (verifier_dir / "reason.txt").write_text(json.dumps(scores, indent=2, sort_keys=True) + "\n")
    print(json.dumps(scores, indent=2, sort_keys=True))
