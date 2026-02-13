#!/bin/bash

#SBATCH --job-name=F_no_retnorm_8_GPU_Bi_ret_12_synth_increase_gamma_along_layers_RetNet
#SBATCH --output=./Job_out/F_no_retnorm_8_GPU_Bi_ret_12_synth_increase_gamma_along_layers_RetNet.out
#SBATCH --gres=gpu:a40:8               # Request 1 A40 GPU
#SBATCH --time=24:00:00                  # Request 24 hours of runtime
#SBATCH --ntasks=1                       # Run a single task
#SBATCH --cpus-per-task=64               # Request 16 CPU cores
#SBATCH --partition=a40                  # Use the A40 partition
#SBATCH --error=./Job_out/F_no_retnorm_8_GPU_Bi_ret_12_synth_increase_gamma_along_layers_RetNet.err

export PYTHONPATH=/home/hpc/iwi5/iwi5295h/RetNetHTR:$PYTHONPATH

cd ..


# Set NCCL timeout to 60 minutes (value is in seconds)
export NCCL_TIMEOUT=3600
# Tell PyTorch to match this timeout (if supported by your version)
export TORCH_DISTRIBUTED_TIMEOUT=3600

# --- DATA STAGING ---
echo "Start staging IAM data to $TMPDIR..."
start_time=$(date +%s)

# 1. Handle IAM Dataset (Move this! It is small but contains many files)
# We assume you made the tar file as discussed.
# If you didn't, change this back to 'cp -r'
cp /home/hpc/iwi5/iwi5295h/IAM/IAM_deslanted.tar $TMPDIR/
tar -xf $TMPDIR/IAM_deslanted.tar -C $TMPDIR/

end_time=$(date +%s)
echo "IAM staging complete. Time taken: $((end_time - start_time)) seconds."
# --------------------

# 2. Handle LMDB Files (The large files)
# We create a directory and copy the files
echo "Copying LMDB files..."
start_time=$(date +%s)

#mkdir -p $TMPDIR/10M_lmdbs
#cp -r /home/vault/iwi5/iwi5295h/10M_lmdbs/* $TMPDIR/10M_lmdbs/

end_time=$(date +%s)
echo "Data staging complete. Time taken: $((end_time - start_time)) seconds."
# ------------------------------------------



export RANK=${SLURM_PROCID}       # Global process rank
export LOCAL_RANK=${SLURM_LOCALID} # Local rank on the node
export WORLD_SIZE=${SLURM_NTASKS}  # Total number of tasks
export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)

# Run your application or script
#torchrun --standalone --nnodes=1 --nproc_per_node=8 train.py --run_name=no_retnorm_8_GPU_Bi_ret_12_synth_increase_gamma_along_layers_RetNet --lmdb_dir=$TMPDIR/10M_lmdbs/ --train_data_dir=$TMPDIR/IAM_deslanted --decoder=Bi_ret_RetNet_Sinusoidal_to_out --batch_size=48 --start_epochs=1 --epochs=1 --lr=0.0001 --weight_decay=1e-03 --lr_scheduler=CosineAnnealingLR --cnn_dropout=0.3 --decoder_dropout=0.3 --img_emb_dropout=0.1 --decoder_emb_dropout=0.1 --p_aug=0.5 --mode=train_inference_recurrent --img_width=2227 --img_height=64 --patch_size=4 --num_channels=1 --embed_dim=768 --d_model=768 --decoder_attention_heads=12 --decoder_ffn_dim=3072 --decoder_depth=12 --cosineRestartEpoch=30 --split=A --weight_init=sd0.02 --bias_init=0 --label_smooth --modeldir=models --feature_extractor=efficientnet_v2_s --gamma_subtracter=0.86 --increase_gamma_along_layers --use_pre_trained_backbone --train_with_synthesized_lmdb --lmdb_path=/home/vault/iwi5/iwi5295h/10M_lmdbs/10M_merged_synthesized_images_part1.lmdb/ --DDP --ret_norm --eval_cycle=1
torchrun --standalone --nnodes=1 --nproc_per_node=8 train.py --run_name=F_no_retnorm_8_GPU_Bi_ret_12_synth_increase_gamma_along_layers_RetNet --train_data_dir=$TMPDIR/IAM_deslanted --decoder=Bi_ret_RetNet_Sinusoidal_to_out --batch_size=16 --start_epochs=1 --epochs=1000 --lr=5e-5 --weight_decay=1e-03 --lr_scheduler=CosineAnnealingLR --cnn_dropout=0.3 --decoder_dropout=0.3 --img_emb_dropout=0.1 --decoder_emb_dropout=0.1 --p_aug=0.5 --mode=train_inference_recurrent --img_width=2227 --img_height=64 --patch_size=4 --num_channels=1 --embed_dim=768 --d_model=768 --decoder_attention_heads=12 --decoder_ffn_dim=3072 --decoder_depth=12 --cosineRestartEpoch=30 --split=A --weight_init=sd0.02 --bias_init=0 --label_smooth --modeldir=models --feature_extractor=efficientnet_v2_s --gamma_subtracter=0.86 --increase_gamma_along_layers --use_pre_trained_backbone --lmdb_path=/home/vault/iwi5/iwi5295h/10M_lmdbs/10M_merged_synthesized_images_part1.lmdb/ --DDP --ret_norm --eval_cycle=1
#torchrun --standalone --nnodes=1 --nproc_per_node=8 train.py --run_name=no_retnorm_8_GPU_Bi_ret_12_synth_increase_gamma_along_layers_RetNet --train_data_dir=$TMPDIR/IAM_deslanted --decoder=Bi_ret_RetNet_Sinusoidal_to_out --batch_size=48 --start_epochs=1 --epochs=1 --lr=0.0001 --weight_decay=1e-03 --lr_scheduler=CosineAnnealingLR --cnn_dropout=0.3 --decoder_dropout=0.3 --img_emb_dropout=0.1 --decoder_emb_dropout=0.1 --p_aug=0.5 --mode=train_inference_recurrent --img_width=2227 --img_height=64 --patch_size=4 --num_channels=1 --embed_dim=768 --d_model=768 --decoder_attention_heads=12 --decoder_ffn_dim=3072 --decoder_depth=12 --cosineRestartEpoch=30 --split=A --weight_init=sd0.02 --bias_init=0 --label_smooth --modeldir=models --feature_extractor=efficientnet_v2_s --gamma_subtracter=0.86 --increase_gamma_along_layers --use_pre_trained_backbone --train_with_synthesized_lmdb --lmdb_path=/home/vault/iwi5/iwi5295h/10M_lmdbs/10M_merged_synthesized_images_part1.lmdb/ --DDP --ret_norm --eval_cycle=1