#!/usr/bin/env bash
set -euo pipefail
set -x

TRAIN_TEST_SPLIT="navtest"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/inspire/hdd/project/agentend2end/xieyuan-24039/hqs/datasets/navsim/maps"
export NAVSIM_EXP_ROOT="/inspire/hdd/project/agentend2end/xieyuan-24039/hqs/datasets/navsim/exp"
export NAVSIM_DEVKIT_ROOT="/inspire/hdd/project/agentend2end/xieyuan-24039/hqs/code/recogdrive"
export OPENSCENE_DATA_ROOT="/inspire/hdd/project/agentend2end/xieyuan-24039/hqs/datasets/navsim"
export PYTHONPATH="/inspire/hdd/project/agentend2end/xieyuan-24039/hqs/code/recogdrive"

CACHE_PATH="/inspire/hdd/project/agentend2end/xieyuan-24039/hqs/datasets/navsim/exp/metric_cache"

echo "NAVSIM_DEVKIT_ROOT: ${NAVSIM_DEVKIT_ROOT}"
echo "NUPLAN_MAPS_ROOT: ${NUPLAN_MAPS_ROOT}"
echo "NAVSIM_EXP_ROOT: ${NAVSIM_EXP_ROOT}"
echo "OPENSCENE_DATA_ROOT: ${OPENSCENE_DATA_ROOT}"
echo "CACHE_PATH: ${CACHE_PATH}"
echo "TRAIN_TEST_SPLIT: ${TRAIN_TEST_SPLIT}"

python "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_metric_caching.py" \
    train_test_split="${TRAIN_TEST_SPLIT}" \
    cache.cache_path="${CACHE_PATH}"
