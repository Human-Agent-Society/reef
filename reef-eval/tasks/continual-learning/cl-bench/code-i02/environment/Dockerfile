FROM pgasawa2/continual-learning-bench:tablib
COPY test.patch /opt/clbench/test.patch
RUN cd /testbed && git reset --hard && git clean -fd \
    && git checkout 4363c435f5f7c99c5adb5d98ba0263f6fecd78e3 \
    && (git apply --verbose /opt/clbench/test.patch \
        || git apply --verbose --reject /opt/clbench/test.patch \
        || patch --batch --fuzz=5 -p1 -i /opt/clbench/test.patch)
WORKDIR /testbed
