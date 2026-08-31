#!/bin/sh
# The benchmark task's Warmup block as the container's entry command: start
# both mock services (their pip install is baked into the image), give them
# time to load the fixtures, then hide the ground truth from the agent.
# The compose healthcheck holds the trial until all of this has happened.
set -eu
export GMAIL_FIXTURES=/tmp_workspace/fixtures/gmail/inbox.json
export CALENDAR_FIXTURES=/tmp_workspace/fixtures/calendar/events.json
python3 /tmp_workspace/mock_services/gmail/server.py &
python3 /tmp_workspace/mock_services/calendar/server.py &
sleep 3
rm -rf /tmp_workspace/fixtures
exec sleep infinity
