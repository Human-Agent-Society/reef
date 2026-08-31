A student will message you about their homework (a math word problem in
`homework/42.txt` under your workspace). Converse with them through the
judge service at `$JUDGE_URL`: `GET /state` returns their current message,
`POST /reply {"text": ...}` sends your answer and returns their reaction.
Help them the way they ask until they say the session is over.

(The bundled harness runs this loop with a Hermes agent whose model calls go
through a Reef training service; see `harness/agent.py`.)
