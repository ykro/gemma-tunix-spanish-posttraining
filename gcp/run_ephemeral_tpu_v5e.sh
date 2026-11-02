#!/usr/bin/env bash
# ==============================================================================
# Ephemeral Cloud TPU v5e Post-Training Runner (Gemma + Flax NNX + Tunix GRPO)
# Project: your own (set PROJECT_ID)
# Guarantee: Zero idle TPU billing via mandatory EXIT trap cleanup
# ==============================================================================
set -euo pipefail

export CLOUDSDK_METRICS_ENVIRONMENT="${CLOUDSDK_METRICS_ENVIRONMENT:+$CLOUDSDK_METRICS_ENVIRONMENT }datacloud.antigravity"

PROJECT_ID="${PROJECT_ID:?Set PROJECT_ID to your Google Cloud project id}"
ZONE="${ZONE:-us-central1-a}"
ACCELERATOR_TYPE="${ACCELERATOR_TYPE:-v5litepod-4}"
RUNTIME_VERSION="${RUNTIME_VERSION:-v2-alpha-tpuv5-lite}"
TPU_NAME="${TPU_NAME:-gemma-tunix-grpo-$(date +%s)}"
GCS_BUCKET="${GCS_BUCKET:-gs://${PROJECT_ID}-tunix-artifacts}"
RUN_ID="run-$(date -u +%Y%m%dT%H%M%SZ)"

cleanup() {
  local exit_code=$?
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] [CLEANUP] Destroying ephemeral TPU VM '${TPU_NAME}' in ${ZONE}..."
  gcloud compute tpus tpu-vm delete "${TPU_NAME}" \
    --project="${PROJECT_ID}" \
    --zone="${ZONE}" \
    --quiet || true
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] [CLEANUP] Ephemeral TPU teardown finished (exit_code=${exit_code})."
}
trap cleanup EXIT INT TERM

echo "======================================================================"
echo " Ephemeral Cloud TPU v5e Post-Training Pipeline"
echo " Project      : ${PROJECT_ID}"
echo " Zone         : ${ZONE}"
echo " Accelerator  : ${ACCELERATOR_TYPE} (${RUNTIME_VERSION})"
echo " TPU VM Name  : ${TPU_NAME}"
echo " Artifact URI : ${GCS_BUCKET}/${RUN_ID}"
echo "======================================================================"

echo "[1/4] Provisioning ephemeral Cloud TPU VM..."
gcloud compute tpus tpu-vm create "${TPU_NAME}" \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}" \
  --accelerator-type="${ACCELERATOR_TYPE}" \
  --version="${RUNTIME_VERSION}" \
  --spot

echo "[2/4] Syncing repository to TPU workers..."
gcloud compute tpus tpu-vm ssh "${TPU_NAME}" \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}" \
  --worker=all \
  --command="mkdir -p ~/gemma-tunix-spanish-posttraining/benchmarks"
gcloud compute tpus tpu-vm scp --recurse \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}" \
  --worker=all \
  ./experiments ./verify_posttraining.py ./pyproject.toml \
  "${TPU_NAME}:~/gemma-tunix-spanish-posttraining/"

echo "[3/4] Executing Gemma + Tunix GRPO post-training & Outside Judge on TPU v5e..."
gcloud compute tpus tpu-vm ssh "${TPU_NAME}" \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}" \
  --worker=all \
  --command="
    set -euo pipefail
    cd ~/gemma-tunix-spanish-posttraining
    python3 -m pip install --upgrade pip
    python3 -m pip install 'jax[tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
    python3 -m pip install flax optax numpy
    python3 experiments/01_ephemeral_tpu_mesh.py
    python3 experiments/02_gemma_nnx_architecture.py
    python3 experiments/03_lora_surgical_injection.py
    python3 experiments/04_grpo_alignment_loop.py
    python3 experiments/05_acid_test_evaluation.py
    python3 verify_posttraining.py
  "

echo "[4/4] Uploading verified benchmark outputs to ${GCS_BUCKET}/${RUN_ID}..."
gcloud compute tpus tpu-vm ssh "${TPU_NAME}" \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}" \
  --worker=0 \
  --command="
    gcloud storage cp -r ~/gemma-tunix-spanish-posttraining/benchmarks '${GCS_BUCKET}/${RUN_ID}/'
  "

echo "[DONE] Pipeline completed. Triggering automatic TPU VM teardown via EXIT trap..."
