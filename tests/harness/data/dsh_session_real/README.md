# Real dsh session fixture

`session.jsonl` is a real dsh 0.1.2-alpha.5 session. It comes from one headless run against a scripted OpenAI-compatible stub. The stub asked for one `bash` call, then answered `READY`. Machine paths in the file and its directory name were replaced with `/work/session-capture`. Nothing else was changed. The ATIF converter tests read it.
