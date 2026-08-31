#!/usr/bin/env bash
set -euo pipefail

ROCKSDB_DIR="${ROCKSDB_DIR:-/app/rocksdb}"
OUT="${1:-/app/solution.patch}"
ALLOWLIST=(
  db/compaction/compaction_picker.cc
  db/compaction/compaction_picker.h
  db/compaction/compaction_picker_level.cc
  db/compaction/compaction_picker_level.h
  db/version_set.cc
)

if [[ ! -d "$ROCKSDB_DIR/.git" ]]; then
  echo "RocksDB checkout not found at $ROCKSDB_DIR" >&2
  exit 2
fi

declare -A allowed=()
for path in "${ALLOWLIST[@]}"; do
  allowed["$path"]=1
done

outside=()
while IFS= read -r -d '' path; do
  [[ ${allowed[$path]+yes} ]] || outside+=("$path")
done < <(
  {
    git -C "$ROCKSDB_DIR" diff --name-only -z HEAD --
    git -C "$ROCKSDB_DIR" ls-files --others --exclude-standard -z
  }
)

if (( ${#outside[@]} )); then
  echo "ERROR: checkout contains changes outside the editable surface:" >&2
  printf '  %q\n' "${outside[@]}" >&2
  exit 3
fi

git -C "$ROCKSDB_DIR" diff --binary --no-color HEAD -- "${ALLOWLIST[@]}" > "$OUT"
python3 /app/validate_solution_patch.py "$OUT"
echo "Wrote $OUT ($(wc -c < "$OUT" | tr -d ' ') bytes)."
