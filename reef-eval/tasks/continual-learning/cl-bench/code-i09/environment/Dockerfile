FROM pgasawa2/continual-learning-bench:tablib
COPY test.patch /opt/clbench/test.patch
RUN cd /testbed && git reset --hard && git clean -fd \
    && git checkout 70775bc922602ae535e0b2e525f6926030334ab8 \
    && (git apply --verbose /opt/clbench/test.patch \
        || git apply --verbose --reject /opt/clbench/test.patch \
        || patch --batch --fuzz=5 -p1 -i /opt/clbench/test.patch)
WORKDIR /testbed
