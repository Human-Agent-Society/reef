---
description: Ask reef to change this harness, or install what it produced (update, install <id>, status)
argument-hint: <what the harness should do> | update | install <release> | status
allowed-tools: Bash(reef-claude:*)
---
The user typed `/reefine` in a Claude Code session that reef serves. reef is the service that evolves this
harness: it takes a request in plain words, writes the change (a skill, a rule, a command, a settings change),
tries it, and publishes it as a release the user installs. Everything below runs through the `reef-claude`
wrapper with the Bash tool; it is on PATH in this session. Report what the wrapper prints, line by line, as it
printed it; never paraphrase a result line, a release id or a link.

The arguments the user typed: $ARGUMENTS

Decide by the arguments:

1. Empty: ask the user in one sentence what the harness should do, and stop.

2. Exactly `status`: run `reef-claude doctor` and show its lines.

3. Exactly `update`: run `reef-claude update`. On exit 0 tell the user the release is installed and a new
   `reef-claude` session runs it. On exit 3 the release requires setup: show the items and, one at a time, ask
   the user for each value or confirmation, then run `reef-claude setup --set NAME=VALUE` for an env item or
   `reef-claude setup --run NAME` for a check; when every item is met run `reef-claude update` again.

4. `install <release>`: run `reef-claude install --release <release>` (the id's first characters suffice).
   This promotes a release that waits for review, then installs it; handle exit 3 as in 3.

5. Anything else is the request. Run, with the Bash tool and a timeout of 1800000 ms:

       reef-claude evolve --wait --timeout 1700 -- <the request, as one quoted argument>

   Quote the request for the shell as written; do not reword it. While it runs reef designs the change,
   writes it, reviews it and tries it, which takes minutes; if Claude Code moves the command to the
   background, tell the user the request is filed and that you will report when it completes, and wait
   for that. Then read the result line:
   - "is published as release X": the change passed and is served. Ask the user whether to install it now;
     on yes run `reef-claude update` (as in 3) and tell them to start a new `reef-claude` session to use it.
   - "is ready as release X ... read it before it runs": the change alters what runs on their machine (a
     settings change with hooks, say) and waits for their review. Give them the page link from the output and
     ask whether they want it installed; only on an explicit yes run `reef-claude install --release X`.
   - "did not pass the checks" or "produced no change": nothing changed; show the reason and suggest how the
     user might rephrase or split the request. Never file a second request on your own: the user decides.
   - exit 2: the step still runs; give the user the "watch it here" link and say `reef-claude doctor` reports
     the result later.

Never run any other command for this task, never edit files of the harness yourself, and never install a
release the user did not ask to install.
