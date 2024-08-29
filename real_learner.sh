source ~/anaconda3/etc/profile.d/conda.sh

conda activate serl

export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2 && \
python main.py "$@" \
    --learner \
    --env LeapGoalReach-State-v0 \
    --exp_name=debug_leap_tactile_pipeline \
    --seed 0 \
    --random_steps 100 \
    --training_starts 1 \
    --utd_ratio 6 \
    --batch_size 384 \
    --eval_period 2000 \
    --encoder_type resnet-pretrained \
    --demo_path peg_insert_20_demos_2023-12-25_16-13-25.pkl \
    --checkpoint_period 1000 \
    --checkpoint_path ./checkpoints/debug
