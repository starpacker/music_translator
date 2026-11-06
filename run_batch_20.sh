#!/usr/bin/env bash
# Batch-regenerate all 20 吹管 scores (OMR -> reconcile -> per-measure-blank render).
# Each 曲子 subdirectory is its own batch.py run, mirroring the existing output layout.
set -u
source /tmp/music_translator_env/bin/activate
cd /liuyutian/music_translator_refresh

LOG=/tmp/batch_20_run.log
: > "$LOG"
echo "[$(date '+%F %T')] start batch (20 scores)" | tee -a "$LOG"

declare -a GROUPS=(小河淌水 渔歌 雨夜花 黑走马)
TOTAL_DONE=0
TOTAL_FAIL=0

for g in "${GROUPS[@]}"; do
  in_dir="input/吹管/$g"
  out_dir="output_fixed/吹管/$g"
  echo "[$(date '+%F %T')] === group $g ===" | tee -a "$LOG"
  python -u batch.py "$in_dir" -o "$out_dir" \
      --mode both --layout single --pitch-mode fixed --fixed-tonic C --gpu off \
      >> "$LOG" 2>&1
  rc=$?
  echo "[$(date '+%F %T')] group $g finished rc=$rc" | tee -a "$LOG"
  if [ $rc -ne 0 ]; then TOTAL_FAIL=$((TOTAL_FAIL+1)); fi
done

echo "[$(date '+%F %T')] DONE. groups_failed=$TOTAL_FAIL" | tee -a "$LOG"
echo "--- generated insitu PDFs ---" | tee -a "$LOG"
find output_fixed/吹管 -name '*.insitu.pdf' -newermt '-10 minutes' | sort | tee -a "$LOG"
