#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root after creating teacher probabilities.
TEACHER_PROBS="${1:-research_outputs/yosm_transformer/lora_seed_42/teacher_train_probabilities.npz}"
for seed in 13 21 42 87 101; do
  python research/lightweight_student.py \
    --teacher_probs "$TEACHER_PROBS" \
    --seed "$seed" \
    --benchmark
done
python research/aggregate_runs.py --root research_outputs/yorubalitesent
