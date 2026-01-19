#!/bin/bash

# Define configs list
configs=(
    "allLayers_bnews_20features"
    "allLayers_bnews_50features"
    "allLayers_bnews_100features"
    "allLayers_bigDataset_20features"
    "allLayers_bigDataset_50features"
    "allLayers_bigDataset_100features"
)

# Iterate through configs and process each one
for config in "${configs[@]}"; do
    python src/train_eval_svc.py --config-name="$config"
done
echo "All experiments completed."
