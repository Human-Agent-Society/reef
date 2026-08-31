FROM pgasawa2/continual-learning-bench:tenacity
COPY test.patch /opt/clbench/test.patch
RUN cd /testbed && git reset --hard && git clean -fd \
    && git checkout cb320c4d80d8af7e8953dbf40224172a06b9b977 \
    && (git apply --verbose /opt/clbench/test.patch \
        || git apply --verbose --reject /opt/clbench/test.patch \
        || patch --batch --fuzz=5 -p1 -i /opt/clbench/test.patch)
WORKDIR /testbed
