#!/usr/bin/env bash
set -euo pipefail
set -x

PROJECT_ROOT=${PROJECT_ROOT:-"/inspire/hdd/project/agentend2end/xieyuan-24039/hqs"}
DATA_ROOT=${DATA_ROOT:-"${PROJECT_ROOT}/datasets/navsim"}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECOGDRIVE_ROOT=${RECOGDRIVE_ROOT:-"$(cd "${SCRIPT_DIR}/../.." && pwd)"}

TRAIN_TEST_SPLIT=${TRAIN_TEST_SPLIT:-navtest}

export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT=${NUPLAN_MAPS_ROOT:-"${DATA_ROOT}/maps"}
export NAVSIM_EXP_ROOT=${NAVSIM_EXP_ROOT:-"${DATA_ROOT}/exp"}
export NAVSIM_DEVKIT_ROOT=${NAVSIM_DEVKIT_ROOT:-"${RECOGDRIVE_ROOT}"}
export OPENSCENE_DATA_ROOT=${OPENSCENE_DATA_ROOT:-"${DATA_ROOT}"}
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

MASTER_PORT=${MASTER_PORT:-63669}
PORT=${PORT:-63665}
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
# CHECKPOINT=${CHECKPOINT:-"${NAVSIM_EXP_ROOT}/exp_baseline_h200/train/train_baseline_fp32_h200/best.ckpt"}
CHECKPOINT=${CHECKPOINT:-"/inspire/hdd/project/agentend2end/xieyuan-24039/hqs/checkpoints/recogdrive/ReCogDrive-2B-IL/ReCogDrive_Diffusion_Planner_2B_IL.ckpt"}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"recogdrive_agent_eval_2b"}
# TORCHRUN_BIN=${TORCHRUN_BIN:-"/opt/conda/envs/recog/bin/torchrun"}
TORCHRUN_BIN=${TORCHRUN_BIN:-"/inspire/hdd/project/agentend2end/xieyuan-24039/hqs/conda_envs/recog/bin/torchrun"}
LOG_FILE=${LOG_FILE:-"logs/eval_recogdrive_2b_h200.log"}

echo "PROJECT_ROOT: ${PROJECT_ROOT}"
echo "DATA_ROOT: ${DATA_ROOT}"
echo "NAVSIM_DEVKIT_ROOT: ${NAVSIM_DEVKIT_ROOT}"
echo "CHECKPOINT: ${CHECKPOINT}"
echo "VLM_PATH: ${VLM_PATH}"

if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "ERROR: CHECKPOINT does not exist: ${CHECKPOINT}" >&2
    exit 1
fi

"${TORCHRUN_BIN}" \
    --nnodes=${NNODES} \
    --nproc_per_node=${GPUS_PER_NODE} \
    --master_port=${MASTER_PORT} \
    "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score_recogdrive.py" \
    train_test_split="${TRAIN_TEST_SPLIT}" \
    agent=recogdrive_agent \
    agent.checkpoint_path="${CHECKPOINT}" \
    agent.vlm_path="${VLM_PATH}" \
    agent.cam_type='single' \
    agent.grpo=False \
    agent.cache_hidden_state=False \
    agent.vlm_type="internvl" \
    agent.dit_type="small" \
    agent.vlm_size="small" \
    agent.sampling_method="ddim" \
    experiment_name="${EXPERIMENT_NAME}" \
    2>&1 | tee "${LOG_FILE}"
