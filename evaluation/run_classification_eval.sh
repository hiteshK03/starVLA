#!/bin/bash
# VLANeXt classification head — LIBERO Spatial evaluation (5-shard parallel)
# Runs model server on GPU 0, 5 LIBERO shards via Docker (CPU-only, OSMesa)
set -e

EVAL_DIR="/mnt/dcgpuval/hkandala/vla-evaluation-harness"
SERVER_CONFIG="configs/model_servers/starvla/vlanext_libero_spatial_classification.yaml"
BENCH_CONFIG="configs/vlanext_starvla_classification_libero_spatial.yaml"
NUM_SHARDS=5

cd "${EVAL_DIR}"

# Source env
source ~/.claude/.env 2>/dev/null || true

echo "=== Starting model server ==="
HIP_VISIBLE_DEVICES=0 uv run vla-eval serve -c "${SERVER_CONFIG}" &
SERVER_PID=$!

# Wait for server to be ready
echo "Waiting for model server to start..."
for i in $(seq 1 120); do
    if python3 -c "import websocket; ws = websocket.create_connection('ws://localhost:8000', timeout=2); ws.close()" 2>/dev/null; then
        echo "Model server ready after ${i}s"
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "ERROR: Model server crashed"
        exit 1
    fi
    sleep 1
done

echo "=== Launching ${NUM_SHARDS}-shard evaluation ==="
PIDS=()
for i in $(seq 0 $((NUM_SHARDS - 1))); do
    echo "Starting shard $i/${NUM_SHARDS}..."
    uv run vla-eval run -c "${BENCH_CONFIG}" \
        --shard-id $i --num-shards ${NUM_SHARDS} &
    PIDS+=($!)
done

# Wait for all shards
echo "Waiting for all shards to complete..."
FAILED=0
for pid in "${PIDS[@]}"; do
    if ! wait $pid; then
        echo "Shard PID $pid failed"
        FAILED=$((FAILED + 1))
    fi
done

# Kill model server
kill $SERVER_PID 2>/dev/null || true

if [ $FAILED -gt 0 ]; then
    echo "ERROR: $FAILED shard(s) failed"
    exit 1
fi

echo "=== Merging results ==="
uv run vla-eval merge -c "${BENCH_CONFIG}"

echo "=== Done ==="
echo "Results in: ${EVAL_DIR}/results/vlanext_starvla_classification_spatial_50ep/"
