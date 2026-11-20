#!/bin/sh

LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH \
torchrun \
  --node_rank="$NODE_RANK" \
  --nnodes="$NUM_NODES"\
  --nproc_per_node="$NUM_GPUS_PER_NODE" \
  --rdzv_id=456 \
  --rdzv_backend=c10d \
  --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
  eval_multiGPU.py
