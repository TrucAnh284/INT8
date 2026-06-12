#!/usr/bin/env bash
# Collect final results from both running experiments and update EXPERIMENTS.md
# Run this after both runs complete.

set -e
SERVER="root@159.48.242.3"
PORT=21418
PW="rri_6wTFDP4joRf65UzW"
SSH="sshpass -p $PW ssh -p $PORT -o StrictHostKeyChecking=no -o PreferredAuthentications=password -o PubkeyAuthentication=no $SERVER"
RSYNC="sshpass -p $PW rsync -az -e 'ssh -p $PORT -o StrictHostKeyChecking=no -o PreferredAuthentications=password -o PubkeyAuthentication=no'"
SPIDER2_LOCAL="/Users/Brian/Desktop/Text2SQL/spider2/spider2_pipeline"
SPIDER1_LOCAL="/Users/Brian/Desktop/Text2SQL/spider1_pipeline"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Collect Results Script"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── Spider 2.0 SC3+2pass results ─────────────────────────────────────────────
echo ""
echo "1. Spider 2.0 SC=3 + 2-pass + fs3 (from server)"

N_PREDS=$($SSH 'ls /root/spider2/spider2_pipeline/output/predictions_fs3_sc3_p2/*.sql 2>/dev/null | wc -l')
echo "   Predictions: $N_PREDS / 135"

if [ "$N_PREDS" -eq "135" ]; then
    echo "   Run COMPLETE — downloading results..."
    eval "$RSYNC $SERVER:/root/spider2/spider2_pipeline/output/results_fs3_sc3_p2.json $SPIDER2_LOCAL/output/results_fs3_sc3_p2.json" 2>/dev/null || true
    eval "$RSYNC $SERVER:/root/spider2/spider2_pipeline/output/results.json $SPIDER2_LOCAL/output/results_fs3_sc3_p2_latest.json" 2>/dev/null || true
    python3 -c "
import json
for f in ['$SPIDER2_LOCAL/output/results_fs3_sc3_p2.json',
          '$SPIDER2_LOCAL/output/results_fs3_sc3_p2_latest.json']:
    try:
        d = json.load(open(f))
        s = d['summary']
        print(f'  {f}: EX={s[\"accuracy\"]:.4f} correct={s[\"correct\"]}/{s[\"total\"]} exec_err={s[\"exec_errors\"]}')
    except: pass
"
else
    echo "   Still running ($N_PREDS/135 done)"
    $SSH 'tail -3 /root/spider2/spider2_pipeline/output/spider2_fs3_sc3_p2.log'
fi

# ── INT8 downstream EX (Spider 1.0) ──────────────────────────────────────────
echo ""
echo "2. INT8 downstream EX — Spider 1.0 sc3+k5+2pass (local)"

INT8_LOG="$SPIDER1_LOCAL/output/run_int8.log"
INT8_RESULTS="$SPIDER1_LOCAL/output/results_int8.json"
INT8_PRED="$SPIDER1_LOCAL/output/predicted_sql_int8.txt"

if [ -f "$INT8_RESULTS" ]; then
    echo "   Run COMPLETE — results:"
    python3 -c "
import json
d = json.load(open('$INT8_RESULTS'))
s = d['summary']
print(f'  EX={s[\"accuracy\"]:.4f} correct={s[\"correct\"]}/{s[\"total\"]} exec_err={s[\"exec_errors\"]}')
fp32_ex = 0.8243
int8_ex = s['accuracy']
print(f'  FP32 EX = 0.8243  INT8 EX = {int8_ex:.4f}  Delta = {int8_ex - fp32_ex:+.4f}')
"
elif [ -f "$INT8_LOG" ]; then
    DONE_LINES=$(grep -c "^\[pipeline\]" "$INT8_LOG" 2>/dev/null || echo 0)
    echo "   Still running (~$DONE_LINES progress lines in log)"
    tail -2 "$INT8_LOG"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
