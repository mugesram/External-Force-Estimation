#!/usr/bin/env bash
# All Tier 1 flights: every scenario x seeds (default 0-4; seed 0 is the tuning seed). One at a time (shared ports).
#   tools/run_tier1_batch.sh [SEEDS...]
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SEEDS=("${@:-0 1 2 3 4}"); [ $# -eq 0 ] && SEEDS=(0 1 2 3 4)
for scen in S1_main S2_hover_only S3_sag_only S4_aggressive S5_payload; do
    for seed in "${SEEDS[@]}"; do
        echo "=== $scen seed $seed $(date +%T)"
        "$ROOT/tools/run_tier1.sh" "$scen" "$seed" 2>&1 | grep -E "vehicle mass|ingest\]|check-bag\] .*sensor|Traceback|Error" | cut -c1-200
    done
done
echo "=== batch done $(date +%T)"
