# The judge sidecar image, built once per run from this directory (run.sh does
# it) and shared by all 72 task images as their FROM line. The tag is the
# content hash of this directory, so a task image can never run a sidecar
# that differs from the sources here.
FROM python:3.12-slim
COPY pyproject.toml personas.py student_server.py /pkg/
RUN pip install --no-cache-dir /pkg && rm -rf /pkg
WORKDIR /judge
