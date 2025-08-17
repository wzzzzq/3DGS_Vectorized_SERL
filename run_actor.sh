export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.5 && \
python train_drq.py "$@" \
    --actor \
    --env=PiperMobileRobot-v0 \
    --exp_name=serl_dev_mobile_robot_test \
    --seed 0 \
    --random_steps 300 \
    --encoder_type resnet-pretrained \
    --debug \
    --actor_queue_size=500 \
    --num_envs=50