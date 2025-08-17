#!/bin/bash

# Evaluate the trained policy
export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2 && \
python eval_policy.py \
    --checkpoint_path $(pwd)/checkpoints \
    --env PiperMobileRobot-v0 \
    --num_episodes 10 \
    --render \
    --encoder_type resnet-pretrained
