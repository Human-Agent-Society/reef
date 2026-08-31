FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends sqlite3 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir pandas numpy scipy
WORKDIR /app
COPY study.db /app/study.db
