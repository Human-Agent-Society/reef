# Real opencode session fixture

`opencode.db` is the session database of a real opencode 1.18.18 run. The run used a scripted OpenAI-compatible stub. The stub asked for one `bash` call, then answered `READY`. The write-ahead log was merged into the file. Machine paths were replaced with `/work/session-capture`. Nothing else was changed. The ATIF converter tests read it.
