# DRetHTR – IAM Fine-tuning & Testing Guide

This guide describes how to install dependencies and run **IAM fine-tuning** and **IAM testing** for the DRetHTR configuration used in our experiments.

---

## 1) Install dependencies

Run the following once (recommended inside a virtualenv/conda env, but `--user` also works):

```bash

conda create -n dret_htr python=3.9.5 -y
conda activate dret_htr

pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.2.1 torchvision==0.17.1

pip install transformers==4.51.3 tokenizers==0.21.1 pandas==2.2.3 scikit-learn==1.6.1 numpy==1.26.4 opencv-python==4.11.0.86 kornia==0.8.1 Pillow==8.4.0 matplotlib==3.4.3 scipy==1.13.1 einops==0.8.1 lmdb==1.6.2 datasets==1.18.4 pyarrow==20.0.0 jiwer==3.1.0
```

If you use Linux : 
```bash
export PYTHONPATH=/Users/chkim/PycharmProjects/:$PYTHONPATH
```
elif you use Windows : 
```powershell
$env:PYTHONPATH = "C:\Users\chkim\PycharmProjects\;" + $env:PYTHONPATH
```

---

## 2) Dataset and ckpt download
https://drive.google.com/drive/folders/1j_P99ZY1O93rc9B6gwd6Ei98nd2Jt5jq?usp=sharing


---

## 3) IAM fine-tuning (SLURM)

Submit the training job via SLURM:
```bash
sbatch DRetHTR_PR/sbatch/F_no_retnorm_8_GPU_Bi_ret_12_synth_increase_gamma_along_layers_RetNet.sh
```
This script handles distributed training (8 GPUs) and logs under the run name: F_no_retnorm_8_GPU_Bi_ret_12_synth_increase_gamma_along_layers_RetNet

You can check the log at `DRetHTR/sbatch/Job_out`

---

## 4) IAM test (single command)

Run the evaluation using the recurrent beam decoding mode:
```bash
python train.py --run_name=F_no_retnorm_8_GPU_Bi_ret_12_synth_increase_gamma_along_layers_RetNet --train_data_dir=dataset\IAM_deslanted --decoder=RetNet_Sinusoidal_to_out --batch_size=16 --start_epochs=1 --epochs=1000 --lr=5e-5 --weight_decay=1e-03 --lr_scheduler=CosineAnnealingLR --cnn_dropout=0.3 --decoder_dropout=0.3 --img_emb_dropout=0.1 --decoder_emb_dropout=0.1 --p_aug=0.5 --mode=test_recurrent_beam_wer --img_width=2227 --img_height=64 --patch_size=4 --num_channels=1 --embed_dim=768 --d_model=768 --decoder_attention_heads=12 --decoder_ffn_dim=3072 --decoder_depth=12 --cosineRestartEpoch=30 --split=A --weight_init=sd0.02 --bias_init=0 --label_smooth --modeldir=models --feature_extractor=efficientnet_v2_s --gamma_subtracter=0.86 --increase_gamma_along_layers --use_pre_trained_backbone --eval_cycle=1 --beam_during_test --load_weight_from_DDP --ret_norm --beam_width=5
```
You will see the result
```bash
test cer : 0.0226 | test wer : 0.0656 | F_no_retnorm_8_GPU_Bi_ret_12_synth_increase_gamma_along_layers_RetNet | time : 131.335
```
## Notes

- **IAM split:** This command uses `--split=A` (IAM Aachen split)
- **Decoding mode:** `--mode=test_recurrent_beam_wer` enables recurrent decoding with beam search and report both cer and wer
- **Paths:** Update `--train_data_dir` and `--lmdb_path` to match your filesystem if needed
- **Model weights:** `--load_weight_from_DDP` assumes the checkpoint was produced by DDP training