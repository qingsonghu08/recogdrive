#!/usr/bin/env bash
set -euo pipefail
set -x

PROJECT_ROOT=${PROJECT_ROOT:-"/inspire/hdd/project/agentend2end/xieyuan-24039/hqs"}
DATA_ROOT=${DATA_ROOT:-"${PROJECT_ROOT}/datasets/navsim"}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECOGDRIVE_ROOT=${RECOGDRIVE_ROOT:-"$(cd "${SCRIPT_DIR}/../.." && pwd)"}

TRAIN_TEST_SPLIT=${TRAIN_TEST_SPLIT:-navtrain}
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT=${NUPLAN_MAPS_ROOT:-"${DATA_ROOT}/maps"}
export NAVSIM_EXP_ROOT=${NAVSIM_EXP_ROOT:-"${DATA_ROOT}/exp_baseline"}
export NAVSIM_DEVKIT_ROOT=${NAVSIM_DEVKIT_ROOT:-"${RECOGDRIVE_ROOT}"}
export OPENSCENE_DATA_ROOT=${OPENSCENE_DATA_ROOT:-"${DATA_ROOT}"}

export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

MASTER_PORT=${MASTER_PORT:-63668}
PORT=${PORT:-63664}
GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-${GPUS}}
NNODES=${NNODES:-1}
export MASTER_PORT=${MASTER_PORT}
export PORT=${PORT}

echo "GPUS: ${GPUS}"
echo "GPUS_PER_NODE: ${GPUS_PER_NODE}"
echo "NNODES: ${NNODES}"

export CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-0}
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}:${PYTHONPATH:-}"

mkdir -p logs

VLM_PATH=${VLM_PATH:-"${PROJECT_ROOT}/checkpoints/recogdrive/ReCogDrive-VLM-2B"}
CACHE_PATH=${CACHE_PATH:-"${DATA_ROOT}/cache/recogdrive_cache_trainval"}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"recogdrive_cache_trainval"}
TORCHRUN_BIN=${TORCHRUN_BIN:-"/inspire/hdd/project/agentend2end/xieyuan-24039/hqs/conda_envs/recog/bin/torchrun"}
LOG_FILE=${LOG_FILE:-"logs/cache_hidden_state_trainval_baseline.log"}

echo "PROJECT_ROOT: ${PROJECT_ROOT}"
echo "DATA_ROOT: ${DATA_ROOT}"
echo "NAVSIM_DEVKIT_ROOT: ${NAVSIM_DEVKIT_ROOT}"
echo "CACHE_PATH: ${CACHE_PATH}"
echo "VLM_PATH: ${VLM_PATH}"

"${TORCHRUN_BIN}" \
    --nnodes=${NNODES} \
    --nproc_per_node=${GPUS_PER_NODE} \
    --master_port=${MASTER_PORT} \
    "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_dataset_caching_multi_node.py" \
    agent=recogdrive_agent \
    experiment_name="${EXPERIMENT_NAME}" \
    agent.cam_type='single' \
    agent.cache_hidden_state=True \
    agent.cache_mode=True \
    agent.vlm_type="internvl" \
    agent.vlm_path="${VLM_PATH}" \
    train_test_split="${TRAIN_TEST_SPLIT}" \
    cache_path="${CACHE_PATH}" \
    2>&1 | tee "${LOG_FILE}"

# nohup bash scripts/cache_dataset/run_caching_recogdrive_hidden_state.sh > logs/cache_hidden_state_trainval.nohup.log 2>&1 &