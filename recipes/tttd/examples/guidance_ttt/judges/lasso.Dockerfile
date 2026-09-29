FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends git g++ libgomp1 ca-certificates && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/wq-will/SimpleTES.git /opt/SimpleTES && \
    git -C /opt/SimpleTES checkout --detach 47d3413da1d85dc24341219d47452d2601e56a57
RUN pip install --no-cache-dir numpy==2.2.6 scipy==1.15.3 scikit-learn==1.6.1 threadpoolctl==3.6.0 pyyaml==6.0.2 psutil==7.0.0
RUN mkdir -p /tmp/eigen /opt/SimpleTES/datasets/numerical_tasks/lasso_path/eigen && \
    tar -xzf /opt/SimpleTES/datasets/numerical_tasks/eigen.tar.gz -C /tmp/eigen && \
    cp -a /tmp/eigen/eigen-3.4.0/. /opt/SimpleTES/datasets/numerical_tasks/lasso_path/eigen/ && rm -rf /tmp/eigen
COPY lasso_worker.py /opt/lasso_worker.py
ENV EVALUATOR_CONCURRENT_PROCESSES=1
WORKDIR /opt/SimpleTES/datasets/numerical_tasks/lasso_path
ENTRYPOINT ["python", "/opt/lasso_worker.py"]
