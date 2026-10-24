import json

THRESHOLDS = (3, 5, 8)
repeat_chain = {"key": None, "count": 0}


def listen(payload, next):
    decision = next()
    key = json.dumps([payload["name"], payload["arguments"]], sort_keys=True, default=str)
    repeat_chain["count"] = repeat_chain["count"] + 1 if key == repeat_chain["key"] else 1
    repeat_chain["key"] = key
    if repeat_chain["count"] in THRESHOLDS:
        reminder = (
            f"You have called {payload['name']} with the same arguments {repeat_chain['count']} times in a row. "
            "Change approach or answer."
        )
        return {**decision, "contexts": [*decision.get("contexts", []), reminder]}
    return decision


NAME = "loop_guard"
EVENT = "post_execute"
