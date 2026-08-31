FROM pgasawa2/continual-learning-bench:tenacity
COPY test.patch /opt/clbench/test.patch
RUN cd /testbed && git reset --hard && git clean -fd \
    && git checkout 21189c74a98552a3efa96e538b9ef2f25668d094 \
    && (git apply --verbose /opt/clbench/test.patch \
        || git apply --verbose --reject /opt/clbench/test.patch \
        || patch --batch --fuzz=5 -p1 -i /opt/clbench/test.patch)
WORKDIR /testbed
