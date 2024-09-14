

export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
python main.py "$@" \
    --actor \
    --env FrankaPegInsert-Vision-v0 \
    --exp_name=debug_leap_tactile_pipeline \
    --seed 0 \
    --random_steps 100 \
    --training_starts 200 \
    --utd_ratio 4 \
    --batch_size 64 \
    --eval_period 2000 \
    --ip 192.168.130.162 \
    --encoder_type resnet-pretrained \
    --render \
    # --checkpoint_path /home/undergrad/code/serl_dev/examples/async_pcb_insert_drq/5x5_20degs_100demos_rand_pcb_insert_bc \
    # --eval_checkpoint_step 20000 \
    # --eval_n_trajs 100 \
    # --batch_size 256 \
