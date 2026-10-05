#!/bin/zsh

set -u

readonly PROJECT_DIR="/Users/shayne/Documents/research/2026_projects/icml_ai_ac"
readonly PYTHON="${PROJECT_DIR}/.venv/bin/python"
readonly RUN_REPORT="${PROJECT_DIR}/data/model_runs/icml_2026_pass1_cheap_ensemble/run.json"
readonly MAX_PASSES="${STAGE02_MAX_PASSES:-5}"

export PYTHONDONTWRITEBYTECODE=1

pass=1
exit_code=1
while (( pass <= MAX_PASSES )); do
  print "[stage02] starting resumable pass ${pass}/${MAX_PASSES}"
  "${PYTHON}" -c \
    'import os, runpy; os.chdir("/Users/shayne/Documents/research/2026_projects/icml_ai_ac"); runpy.run_module("icml_ai_ac.cli", run_name="__main__")' \
    score-pass1-ensemble \
    --manifest data/metadata/icml_2026_scoring_anonymized.jsonl \
    --out-dir data/model_runs/icml_2026_pass1_cheap_ensemble \
    --model-preset production_2026_v2 \
    --model-workers 4 \
    --class-path data/metadata/icml_2026_contribution_classes_randomized.jsonl \
    --strategy class_round_robin \
    --batch-size 8 \
    --partitions 2 \
    --timeout 180 \
    --retries 2 \
    --backoff 2 \
    --delay 0.25 \
    --aggregate-out data/scores/icml_2026_pass1_cheap_ensemble_signal.jsonl \
    --aggregate-report data/scores/icml_2026_pass1_cheap_ensemble_signal.report.json
  exit_code=$?
  if (( exit_code == 0 )); then
    print "[stage02] completed successfully"
    exit 0
  fi
  if [[ ! -f "${RUN_REPORT}" ]]; then
    print -u2 "[stage02] run failed before producing a report; stopping"
    exit "${exit_code}"
  fi
  blocked="$("${PYTHON}" -c '
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
print(any(
    isinstance(result, dict) and result.get("blocked_reason")
    for result in report.get("results", [])
))
' "${RUN_REPORT}")"
  if [[ "${blocked}" == "True" ]]; then
    print -u2 "[stage02] blocking provider error recorded; stopping"
    exit "${exit_code}"
  fi
  (( pass += 1 ))
  if (( pass <= MAX_PASSES )); then
    print "[stage02] non-blocking incomplete batches remain; resuming in 5 seconds"
    sleep 5
  fi
done

print -u2 "[stage02] incomplete after ${MAX_PASSES} bounded passes; stopping"
exit "${exit_code}"
