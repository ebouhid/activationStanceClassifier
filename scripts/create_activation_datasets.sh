#!/bin/bash

mkdir -p data

configs=( # We can use only a subset of configs to create activation datasets
    "allLayers_bnews_20features"
    "allLayers_bigDataset_20features"
)

for config in "${configs[@]}"; do
    python src/extract_activations.py --config-name="$config"
done
echo "Activation datasets created."