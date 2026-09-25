# The agent's container: the grid runner alone. No scoring lives here; the
# runner reaches the Reef service on the host and writes its record for the
# verifier.
FROM python:3.12-slim
RUN pip install --no-cache-dir "reef-client>=0.2.0"
WORKDIR /opt/grid
COPY grid.py /opt/grid/
