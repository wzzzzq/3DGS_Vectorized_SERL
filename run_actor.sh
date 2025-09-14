export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=1 && \
python train_drq.py "$@" \
    --actor \
    --env=PiperMobileRobot-v0 \
    --exp_name=serl_dev_mobile_robot_test \
    --seed 0 \
    --random_steps 500 \
    --encoder_type small \
    --debug \
    --actor_queue_size=300 \
    --num_envs=20