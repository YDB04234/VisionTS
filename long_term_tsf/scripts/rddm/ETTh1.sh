export CUDA_VISIBLE_DEVICES=0,1,2,3
export CKPT_DIR="../ckpt/"
export VM_ARCH="mae_base"
export CONTEXT_LEN=96
export PERIODICITY=24
export ALIGN_CONST=0.4
export NORM_CONST=0.4

for PRED_LEN in 96 192 336 720; do
    python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --model ResidualDiffusion \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --save_dir save/ETTh1_$PRED_LEN \
    --model_id ResidualDiffusion_ETTh1_$PRED_LEN \
    --data ETTh1 \
    --features M \
    --train_epochs 0 \
    --vm_arch $VM_ARCH \
    --vm_ckpt $CKPT_DIR \
    --seq_len $CONTEXT_LEN \
    --periodicity $PERIODICITY \
    --pred_len $PRED_LEN \
    --norm_const $NORM_CONST \
    --align_const $ALIGN_CONST
done;