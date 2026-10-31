#!/bin/bash
# The verifier of a converted TMax task: the reward is the fraction of the tests in test_outputs.py that pass,
# written to /logs/verifier/reward.txt. Reimplemented from AutoEnvScaling's managed verifier (Apache 2.0) on
# pytest's own JUnit report, so it needs only pytest, which the image installs at build time.
#
# It fails closed: a missing or unreadable report, or no test, scores 0. A skipped test never counts as passed.
# Reports and rewards left in the verifier's directory are removed first (Harbor prefers reward.json to
# reward.txt). pytest's config file, rootdir and conftest search stop at the tests, so a config file or conftest.py
# above them cannot change the collection, and no installed plugin loads on its own.
# TMAX_TESTS_DIR and TMAX_VERIFIER_DIR move the two directories for this script's own test.
set -u
TESTS_DIR="${TMAX_TESTS_DIR:-/tests}"
VERIFIER_DIR="${TMAX_VERIFIER_DIR:-/logs/verifier}"
mkdir -p "$VERIFIER_DIR"
rm -rf "$VERIFIER_DIR/junit.xml" "$VERIFIER_DIR/reward.json" "$VERIFIER_DIR/reward.txt"
echo 0 > "$VERIFIER_DIR/reward.txt"

config=""
for name in pytest.ini tox.ini setup.cfg pyproject.toml; do
  if [ -f "$TESTS_DIR/$name" ]; then config="$TESTS_DIR/$name"; break; fi
done
if [ -z "$config" ]; then
  # pytest reads -c by its extension and refuses a file without one, so the empty config is a pytest.ini.
  config="$(mktemp -d)/pytest.ini"
  printf '[pytest]\n' > "$config"
fi
unset PYTEST_ADDOPTS PYTEST_PLUGINS
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

cd "$TESTS_DIR" || exit 0
python3 -m pytest -c "$config" --rootdir "$TESTS_DIR" --confcutdir "$TESTS_DIR" -p no:cacheprovider \
  test_outputs.py -q --tb=short --junitxml="$VERIFIER_DIR/junit.xml"

python3 - "$VERIFIER_DIR" <<'PY'
import sys
import xml.etree.ElementTree as ElementTree

verifier_dir = sys.argv[1]
fraction = 0.0
try:
    root = ElementTree.parse(f"{verifier_dir}/junit.xml").getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    total = int(suite.get("tests", 0))
    not_passed = int(suite.get("failures", 0)) + int(suite.get("errors", 0)) + int(suite.get("skipped", 0))
    fraction = max(0.0, (total - not_passed) / total) if total else 0.0
except Exception:
    fraction = 0.0
with open(f"{verifier_dir}/reward.txt", "w") as reward:
    reward.write(f"{fraction:.4f}\n")
print("reward:", fraction)
PY
