FROM pgasawa2/continual-learning-bench:tablib
COPY test.patch /opt/clbench/test.patch
RUN cd /testbed && git reset --hard && git clean -fd \
    && git checkout 1975024f95ad962cd7b3e0a5626b37b16effc9db \
    && (git apply --verbose /opt/clbench/test.patch \
        || git apply --verbose --reject /opt/clbench/test.patch \
        || patch --batch --fuzz=5 -p1 -i /opt/clbench/test.patch)
WORKDIR /testbed
