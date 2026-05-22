#!/usr/bin/env bash
set -euo pipefail
set -x

PROJECT_ROOT=${PROJECT_ROOT:-"/inspire/hdd/project/agentend2end/xieyuan-24039/hqs"}
DATA_ROOT=${DATA_ROOT:-"${PROJECT_ROOT}/datasets/navsim"}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECOGDRIVE_ROOT=${RECOGDRIVE_ROOT:-"$(cd "${SCRIPT_DIR}/../.." && pwd)"}

export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT=${NUPLAN_MAPS_ROOT:-"${DATA_ROOT}/maps"}
export NAVSIM_EXP_ROOT=${NAVSIM_EXP_ROOT:-"${DATA_ROOT}/exp"}
export NAVSIM_DEVKIT_ROOT=${NAVSIM_DEVKIT_ROOT:-"${RECOGDRIVE_ROOT}"}
export OPENSCENE_DATA_ROOT=${OPENSCENE_DATA_ROOT:-"${DATA_ROOT}"}
TRAIN_TEST_SPLIT=${TRAIN_TEST_SPLIT:-navtrain}

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

CACHE_PATH=${CACHE_PATH:-"${DATA_ROOT}/cache/recogdrive_cache_trainval"}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"train_baseline_fp32_h200"}
OUTPUT_DIR=${OUTPUT_DIR:-"${NAVSIM_EXP_ROOT}/train/${EXPERIMENT_NAME}"}
TORCHRUN_BIN=${TORCHRUN_BIN:-"/inspire/hdd/project/agentend2end/xieyuan-24039/hqs/conda_envs/recog/bin/torchrun"}
LOG_FILE=${LOG_FILE:-"logs/train_baseline_h200.log"}

if [[ ! -d "${CACHE_PATH}" ]]; then
    echo "ERROR: CACHE_PATH does not exist: ${CACHE_PATH}" >&2
    exit 1
fi

# echo "PROJECT_ROOT: ${PROJECT_ROOT}"
# echo "DATA_ROOT: ${DATA_ROOT}"
# echo "NAVSIM_DEVKIT_ROOT: ${NAVSIM_DEVKIT_ROOT}"
# echo "CACHE_PATH: ${CACHE_PATH}"
# echo "OUTPUT_DIR: ${OUTPUT_DIR}"

"${TORCHRUN_BIN}" \
    --nnodes=${NNODES} \
    --nproc_per_node=${GPUS_PER_NODE} \
    --master_port=${MASTER_PORT} \
    "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_recogdrive_accelerate.py" \
    agent=recogdrive_agent \
    agent.lr=1e-4 \
    agent.grpo=False \
    agent.cam_type='single' \
    agent.cache_hidden_state=True \
    agent.cache_mode=False \
    agent.vlm_type="internvl" \
    agent.dit_type="small" \
    agent.vlm_size="small" \
    agent.sampling_method="ddim" \
    agent.train_backbone=False \
    trainer.params.max_epochs=200 \
    dataloader.params.batch_size=128 \
    deepspeed.train_micro_batch_size_per_gpu=128 \
    dataloader.params.num_workers=12 \
    deepspeed.zero_optimization.stage=1 \
    '~deepspeed.zero_optimization.offload_optimizer' \
    trainer.params.num_nodes=${NNODES} \
    trainer.params.devices=${GPUS_PER_NODE} \
    trainer.params.check_val_every_n_epoch=10 \
    use_deepspeed=True \
    deepspeed.bf16.enabled=False \
    deepspeed.fp16.enabled=True \
    experiment_name="${EXPERIMENT_NAME}" \
    output_dir="${OUTPUT_DIR}" \
    train_test_split="${TRAIN_TEST_SPLIT}" \
    cache_path="${CACHE_PATH}" \
    use_cache_without_dataset=True \
    force_cache_computation=False \
    2>&1 | tee "${LOG_FILE}"

# nohup bash scripts/training_accelerate/run_recogdrive_train_multi_node_2b.sh > logs/train_baseline_h200.nohup.log2>&1 &
