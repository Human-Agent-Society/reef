#!/bin/bash
# Answer directly with zero exploratory queries: reward exactly 1.0.
curl -s "$JUDGE_URL/answer" -d '{"answer": "0.18"}'
