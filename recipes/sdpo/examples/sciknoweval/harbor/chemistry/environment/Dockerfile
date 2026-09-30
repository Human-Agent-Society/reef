# The agent's container: the stage runner and the reference's Chemistry split.
# No scoring service lives here; the runner reaches the Reef service on the
# host, scores with the reference's own rule, and writes the curve for the
# verifier.
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/* \
    && git clone --quiet https://github.com/lasgroup/SDPO /opt/sdpo-reference \
    && git -C /opt/sdpo-reference checkout --quiet 7c457fc1b1f636ae794eb0362ba37d4743b06fbc \
    && pip install --no-cache-dir "reef-client>=0.2.0"
WORKDIR /opt/chemistry
COPY chemistry.py stage.py /opt/chemistry/
