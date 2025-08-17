export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.8 && \
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false --xla_gpu_triton_gemm_any=false" && \
python train_drq.py "$@" \
    --learner \
    --env=PiperMobileRobot-v0 \
    --exp_name=serl_dev_mobile_robot_test \
    --seed 0 \
    --training_starts 300 \
    --critic_actor_ratio 2 \
    --encoder_type  resnet-pretrained\
    --batch_size 32 \
    --checkpoint_period 100000 \
    --checkpoint_path $(pwd)/checkpoints \
    --debug
