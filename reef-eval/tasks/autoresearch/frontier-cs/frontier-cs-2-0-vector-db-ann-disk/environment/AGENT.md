# Frontier-CS 2.0 Submission Workflow

Submissions are asynchronous. `bash /app/submit.sh` snapshots the current solution,
adds it to the judge queue, and returns immediately with a submission UUID.

Useful commands:

```bash
bash /app/submit.sh
bash /app/submissions.sh
bash /app/wait_submission.sh <submission_uuid>
bash /app/cancel_submission.sh <submission_uuid>
```

The judge evaluates one queued submission at a time. Continue improving your
solution while a queued or running submission is being scored. Cancellation
works for queued submissions and for running async submissions; cancelling a
running submission stops the current evaluation. The queue accepts a small
number of queued/running submissions at once; if it is full, wait for an
existing submission to finish or cancel one.

The final verifier can reuse the best completed iterative submission, so submit
early and often enough to get feedback, but avoid flooding the queue.
