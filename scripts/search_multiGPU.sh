#!/bin/sh

# Reverse-augmentation knobs for beam_predict_multiGPU_v2.py:
#   REVERSE_AUG=1    split SAMPLE_SIZE across forward and reverse predictions
#   REVERSE_SIGMA=x  use x as the reverse-direction noising sigma
#   REVERSE_FLOW=1   run the base integration direction in reverse
#
# Example:
#   REVERSE_AUG=1 REVERSE_SIGMA=0.25 python beam_predict_multiGPU_v2.py
#
# torchrun \
#   --node_rank="$NODE_RANK" \
#   --nnodes="$NUM_NODES"\
#   --nproc_per_node="$NUM_GPUS_PER_NODE" \
#   --rdzv-id=456 \
#   --rdzv-backend=c10d \
#   --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
#   beam_predict_multiGPU.py

python beam_predict_multiGPU_v2.py
