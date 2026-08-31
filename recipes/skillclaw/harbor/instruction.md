Director Chen wants me to set up a "Q2 Product Review" meeting this week, before Friday. 90 minutes, with Li Wei, Zhang Min, and Wang Fang. He should've sent me an email about it — check my inbox for the details.

Can you handle the scheduling? Coordinate with everyone, find a time that works, and get it on the calendar. Let Director Chen know once it's all confirmed.

---

The mailbox and calendar are mock services already running in this
environment: Gmail at `http://localhost:9100`, Calendar at
`http://localhost:9101`. `/tmp_workspace/SKILL.md` documents every endpoint;
send JSON bodies over POST, for example:

    curl -s -X POST http://localhost:9100/gmail/messages \
        -H 'Content-Type: application/json' -d '{}'

The services inject transient 429/500 errors and slow responses on purpose;
retry on failure. When the scheduling is done, write a short summary of what
you did and what was booked to `/tmp_workspace/results/results.md`.
