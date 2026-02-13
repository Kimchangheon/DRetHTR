import torch.nn as nn
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

import os
import warnings
import gc

import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from timeit import default_timer as timer
import re

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFD, Sequence

from dataset.IAMDataset import IAMDataset, ImageTxtProcessor
from dataset.IAMDataset_lmdb import LMDBDataset
from helper import *
from traineval.trainer_evaluator import *
from model.RetNetHTR import DRetHTR
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import glob


from datetime import timedelta

import threading
import time
import subprocess
import os
import signal


warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*compute_measures.*")


def parse_args():
    parser = argparse.ArgumentParser(description='Train Retnet2Retnet with specified parameters.')
    # Add all the required arguments here
    parser.add_argument('--run_name', type=str, default="R32R3_CNN", help='Name of the run')
    parser.add_argument('--train_with_synthesized_lmdb', action='store_true', default=False, help='train using synthesized data when in single processing')
    parser.add_argument('--base_path', type=str, default="/home/hpc/iwi5/iwi5295h/RetNetHTR/DRetHTR", help='base path')
    parser.add_argument('--lmdb_path', type=str, default="/home/vault/iwi5/iwi5295h/10M_lmdbs/10M_merged_synthesized_images_part1.lmdb/", help='lmdb dir for single processing')
    parser.add_argument('--lmdb_dir', type=str, default="/home/vault/iwi5/iwi5295h/10M_lmdbs/", help='lmdb dir for DDP')
    parser.add_argument('--do_train_valid_test', action='store_true', default=False, help='Train Valid Test for parameter search ')
    parser.add_argument('--mode', type=str, default="train_inference_recurrent", help='Mode of operation')
    parser.add_argument('--train_data_dir', type=str, default="/home/hpc/iwi5/iwi5295h/IAM/IAM_deslanted", help='Training data directory')
    parser.add_argument('--decoder', type=str, default="RetNet3", help='Decoder type')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--start_epochs', type=int, default=1, help='Starting epoch number')
    parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--cosineRestartEpoch', type=int, default=30, help='cosineRestartEpoch')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay rate')
    parser.add_argument('--lr_scheduler', type=str, default="CosineAnnealingLR", help='Learning rate scheduler type')
    parser.add_argument('--cnn_dropout', type=float, default=0.3, help='Dropout rate for encoder CNN (feature extractor)')
    parser.add_argument('--decoder_dropout', type=float, default=0.3, help='Dropout rate for decoder')
    parser.add_argument('--img_emb_dropout', type=float, default=0.1, help='Dropout rate for encoder embeddings')
    parser.add_argument('--decoder_emb_dropout', type=float, default=0.1, help='Dropout rate for decoder embeddings')
    parser.add_argument('--p_aug', type=float, default=0.5, help='Probability of augmentation')
    parser.add_argument('--img_width', type=int, default=2227, help='Image width')
    parser.add_argument('--img_height', type=int, default=64, help='Image height')
    parser.add_argument('--patch_size', type=int, default=4, help='Patch size')
    parser.add_argument('--num_channels', type=int, default=1, help='Number of channels in input images')
    parser.add_argument('--embed_dim', type=int, default=128, help='Dimension of the embeddings')
    parser.add_argument('--d_model', type=int, default=128, help='Dimension of the model')
    parser.add_argument('--decoder_attention_heads', type=int, default=4, help='Number of attention heads in decoder')
    parser.add_argument('--decoder_ffn_dim', type=int, default=512, help='Dimension of the feedforward network in the decoder')
    parser.add_argument('--decoder_depth', type=int, default=3, help='Number of layers in the decoder')
    parser.add_argument('--split', type=str, default="A", help='split type (A(achen), B, C, D, Random, A_expand)')
    parser.add_argument('--weight_init', type=str, default="sd0.02", help='Weight initialization method (None, He, sd0.02)')
    parser.add_argument('--bias_init', type=float, default=0.0, help='Bias initialization value (0.01, 0)')
    parser.add_argument('--patch_order', type=int, default=0, help='Patch order (0 or 1, 0 is left bottom to top (column first), 1 is left top to right bottom (row first))')
    parser.add_argument('--image_padding', action='store_false', default=True, help='Enable image padding')
    parser.add_argument('--label_smooth', action='store_true', default=True, help='Enable label smoothing')
    parser.add_argument('--modeldir', type=str, default="models", help='Model directory (models, models2, models3)')
    parser.add_argument('--feed_pad_positions', action='store_true', default=False, help='Feed pad positions to the model')
    parser.add_argument('--feature_extractor', type=str, default="Shallow_CNN", help='Feature extractor type (Patch_embedding, Shallow_CNN, ResNet50)')
    parser.add_argument('--feed_img_pad_ratios', action='store_true', default=False, help='Feed img pad ratios to the model')
    parser.add_argument('--D_norm', action='store_false', default=True, help='Remove D_norm')
    parser.add_argument('--ret_norm', action='store_false', default=True, help='Remove ret_norm')
    parser.add_argument('--gamma_subtracter', type=float, default=0.0, help='subtract value to gamma to make it intense')
    parser.add_argument('--beam_width', type=int, default=10, help='Beam_width for beam search')
    parser.add_argument('--retnorm_inference', action='store_true', default=False, help='retnorm during inference ')
    parser.add_argument('--eval_cycle', type=float, default=20, help='do evaluation in every 20 cycle')
    parser.add_argument('--eval_cycle_epoch', type=float, default=1, help='do evaluation in every 20 cycle')
    parser.add_argument('--various_gamma_in_heads', action='store_true', default=False, help='[0.1088, 0.2317, 0.3545, 0.4774, 0.6002, 0.7231, 0.8459, 0.9688] various_gamma_in_heads')
    parser.add_argument('--increase_gamma_along_layers', action='store_true', default=False, help='[0.1088 ~ 0.9688] increase gamma along layers')
    parser.add_argument('--bpe', action='store_true', default=False, help='Byte pair encoding')
    parser.add_argument('--bpe_size', type=int, default=8000, help='vocab size for byte pair encoding')
    parser.add_argument('--gpt2_tokenizer', action='store_true', default=False, help='use gpt2_tokenizer')
    parser.add_argument('--print_eval', action='store_true', default=False, help='print during decoding')
    parser.add_argument('--use_pre_trained_backbone', action='store_true', default=False, help='print during decoding')
    parser.add_argument('--DDP', action='store_true', default=False, help='DistributedDataParallel')
    parser.add_argument('--load_weight_from_non_DDP', action='store_true', default=False, help='load_weight_from_non_DDP')
    parser.add_argument('--load_weight_from_DDP', action='store_true', default=False, help='load_weight_from_non_DDP')
    parser.add_argument('--skip_emb_generator_weights', action='store_true', default=False, help='load_weight_from_non_DDP')
    parser.add_argument('--beam_during_test', action='store_true', default=False, help='load_weight_from_non_DDP')

    return parser.parse_args()
args = parse_args()

do_train_valid_test = args.do_train_valid_test


import os, time, signal, subprocess

def gpu_idle_watchdog(
    threshold=5,
    idle_minutes=60,          # 1 hour default
    poll_interval=60,
    only_slurm_gpus=True,     # monitor only GPUs in CUDA_VISIBLE_DEVICES
):
    """
    Terminate the job if ANY *single* GPU stays below `threshold`% utilization
    for `idle_minutes` minutes continuously.

    - Tracks per-GPU idle streaks.
    - Optionally restricts to GPUs assigned to the job via CUDA_VISIBLE_DEVICES.

    threshold: utilization percent (0-100)
    idle_minutes: how long a GPU can stay idle continuously before killing job
    poll_interval: seconds between checks
    only_slurm_gpus: if True, only monitor GPUs listed in CUDA_VISIBLE_DEVICES
    """
    job_id = os.environ.get("SLURM_JOB_ID", None)
    max_idle_checks = max(1, int((idle_minutes * 60) // poll_interval))

    # Decide which GPU indices to query
    # If CUDA_VISIBLE_DEVICES="3,5,7" we monitor physical GPUs [3,5,7].
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if only_slurm_gpus and visible.strip():
        try:
            gpu_ids = [int(x) for x in visible.replace(" ", "").split(",") if x != ""]
        except Exception:
            gpu_ids = None  # fallback to all
    else:
        gpu_ids = None

    # Per-GPU idle counters (initialized after first query once we know GPU count)
    idle_counts = None
    last_utils = None

    scope_msg = f"CUDA_VISIBLE_DEVICES={visible}" if (only_slurm_gpus and visible.strip()) else "ALL GPUs on node"
    print(
        f"[GPU watchdog] Started (threshold={threshold}%, idle_minutes={idle_minutes}, "
        f"poll_interval={poll_interval}s, scope={scope_msg}).",
        flush=True
    )

    while True:
        try:
            # Query both index and utilization so we can map correctly
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                encoding="utf-8"
            )

            rows = []
            for line in out.strip().splitlines():
                if not line.strip():
                    continue
                # e.g. "0, 12"
                parts = [p.strip() for p in line.split(",")]
                if len(parts) != 2:
                    continue
                idx = int(parts[0])
                util = int(parts[1])
                rows.append((idx, util))

            if not rows:
                return  # no GPUs found

            # Filter to GPUs assigned to job if requested
            if gpu_ids is not None:
                rows = [(i, u) for (i, u) in rows if i in gpu_ids]
                if not rows:
                    print("[GPU watchdog] No GPUs matched CUDA_VISIBLE_DEVICES; exiting watchdog.", flush=True)
                    return

            # Initialize counters
            if idle_counts is None:
                idle_counts = {i: 0 for (i, _) in rows}
                last_utils = {i: 0 for (i, _) in rows}

            # Update per-GPU idle streaks
            for (i, util) in rows:
                last_utils[i] = util
                if util < threshold:
                    idle_counts[i] += 1
                else:
                    idle_counts[i] = 0  # reset streak if this GPU is active again

            # Check if any single GPU exceeded the idle limit
            offenders = [i for i, c in idle_counts.items() if c >= max_idle_checks]
            if offenders:
                offender_str = ", ".join(
                    f"GPU{i} (util={last_utils[i]}%, idle_checks={idle_counts[i]}/{max_idle_checks})"
                    for i in offenders
                )
                msg = (
                    f"[GPU watchdog] Detected continuously idle GPU(s) below {threshold}% "
                    f"for {idle_minutes} minutes: {offender_str}. Terminating job."
                )
                print(msg, flush=True)

                if job_id is not None:
                    try:
                        subprocess.call(["scancel", job_id])
                    except Exception as e:
                        print(f"[GPU watchdog] scancel failed: {e}, killing PID instead.", flush=True)

                os.kill(os.getpid(), signal.SIGTERM)
                return

        except Exception as e:
            print(f"[GPU watchdog] Error while checking GPU utilization: {e}", flush=True)

        time.sleep(poll_interval)

if args.DDP :
    # Setup Distributed Training (DDP) if available
    print("Set up Distributed Training (DDP)")
    def setup_ddp():
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            rank = int(os.environ['RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            torch.cuda.set_device(local_rank)
            dist.init_process_group(backend="nccl", init_method="env://", timeout=timedelta(hours=1))
            return rank, world_size, local_rank
        else:
            return 0, 1, 0  # default values for non-distributed run

    rank, world_size, local_rank = setup_ddp()

    if world_size > 1:
        print(f"world_size is more than 1, cuda:{local_rank} is assigned as device")
        device = torch.device(f"cuda:{local_rank}")
    else:
        print(f"world_size is less than 1, single cuda is assigned as device")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}, Rank: {rank}, World Size: {world_size}")

    num_cpus = min(os.cpu_count() // world_size, 16) if world_size > 1 else min(os.cpu_count(), 16)

    if device == "cpu":
        num_cpus = 1
    elif world_size > 1:
        torch.multiprocessing.set_sharing_strategy('file_system')

    if rank == 0:  # Only print from the main process
        print(f"Number of CPUs used per process: {num_cpus}, Total CPUs: {os.cpu_count()}, World Size: {world_size}")

else :
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    num_cpus = min(os.cpu_count(), 16)
    if device == "cpu":
        num_cpus = 1
    else:
        torch.multiprocessing.set_sharing_strategy('file_system')
    print("Number of CPUs used:", num_cpus)


# After args = parse_args() and device setup, before training loop:
# Make sure rank is defined. For non-DDP runs, just set rank = 0.
if not args.DDP:
    rank = 0

# Start GPU watchdog only on the main process
if (not args.DDP) or (args.DDP and rank == 0):
    watchdog_thread = threading.Thread(
        target=gpu_idle_watchdog,
        kwargs=dict(threshold=5, idle_minutes=60, poll_interval=60),
        daemon=True,
    )
    watchdog_thread.start()
    print(f"[Rank {rank}] GPU watchdog started (single instance).", flush=True)
else:
    # Optional: one-line info so you know others are quiet
    print(f"[Rank {rank}] GPU watchdog disabled on this rank (handled by rank 0).",
          flush=True)

num_cpus =4
file_path = os.path.join(args.train_data_dir, 'img_transcription.txt')
data = []
with open(file_path, 'r') as file:
    for line in file:
        split_line = line.strip().split('\t')
        if len(split_line) == 2:
            data.append(split_line)
df = pd.DataFrame(data, columns=['file_name', 'text'])
print(df.head())
print("df len : ",len(df))


def read_filenames(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() + '.png' for line in file.readlines()]

valid_df = None  # <-- NEW (so it exists in all cases)

if args.split in ['A', 'B', 'C', 'D']:
    train_filenames = read_filenames(os.path.join('./', f'Splitting/IAM-{args.split}/train.txt'))
    valid_filenames = read_filenames(os.path.join('./', f'Splitting/IAM-{args.split}/valid.txt'))
    test_filenames  = read_filenames(os.path.join('./', f'Splitting/IAM-{args.split}/test.txt'))

    train_df = df[df['file_name'].isin(train_filenames)]
    valid_df = df[df['file_name'].isin(valid_filenames)]
    test_df  = df[df['file_name'].isin(test_filenames)]

    # OLD behavior: merge train+valid into train if not doing 3-way
    if not do_train_valid_test:
        train_df = pd.concat([train_df, valid_df], ignore_index=True)
        valid_df = None

elif args.split == 'Random':
    if do_train_valid_test:
        # Example: 80/10/10 (adjust as you like)
        train_df, tmp_df = train_test_split(df, test_size=0.2, random_state=42)
        valid_df, test_df = train_test_split(tmp_df, test_size=0.5, random_state=42)
    else:
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

elif args.split == "A_expand":
    test_filenames = read_filenames('./Splitting/IAM-A/test.txt')
    test_df = df[df['file_name'].isin(test_filenames)]
    base_train_df = df[~df['file_name'].isin(test_filenames)]

    if do_train_valid_test:
        train_df, valid_df = train_test_split(base_train_df, test_size=0.1, random_state=42)
    else:
        train_df = base_train_df
        valid_df = None

elif args.split in ["Bentham", "RIMES"]:
    train_filenames = read_filenames(os.path.join('./', f'Splitting/{args.split}/train.txt'))
    valid_filenames = read_filenames(os.path.join('./', f'Splitting/{args.split}/valid.txt'))
    test_filenames  = read_filenames(os.path.join('./', f'Splitting/{args.split}/test.txt'))

    train_df = df[df['file_name'].isin(train_filenames)]
    valid_df = df[df['file_name'].isin(valid_filenames)]
    test_df  = df[df['file_name'].isin(test_filenames)]

    if not do_train_valid_test:
        train_df = pd.concat([train_df, valid_df], ignore_index=True)
        valid_df = None

elif args.split == "READ2016":
    train_filenames = read_filenames(os.path.join('./', f'Splitting/{args.split}/train.txt'))
    test_filenames  = read_filenames(os.path.join('./', f'Splitting/{args.split}/test.txt'))

    base_train_df = df[df['file_name'].isin(train_filenames)]
    test_df = df[df['file_name'].isin(test_filenames)]

    if do_train_valid_test:
        train_df, valid_df = train_test_split(base_train_df, test_size=0.1, random_state=42)
    else:
        train_df = base_train_df
        valid_df = None

else:
    raise ValueError("Invalid value for args.split.")


if do_train_valid_test and valid_df is not None:
    df = pd.concat([train_df, valid_df, test_df], ignore_index=True)
else:
    df = pd.concat([train_df, test_df], ignore_index=True)

if rank == 0:
    print(df.head())

if args.p_aug == 0.0 :
    do_aug =False
else :
    do_aug = True

if args.bpe and args.gpt2_tokenizer :
    # Disable parallelism warning from tokenizers
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Load the GPT‑2 tokenizer from Hugging Face
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # GPT‑2 does not have certain special tokens by default, so we add them.
    # (Note: you can choose which tokens to add based on your training setup.)
    special_tokens = {
        "bos_token": "<BOS>",
        "eos_token": "<EOS>",
        "pad_token": "<PAD>",
        "unk_token": "<UNK>"
    }
    tokenizer.add_special_tokens(special_tokens)

    # Print vocabulary size (after adding special tokens)
    vocab_size = len(tokenizer.get_vocab())
    print("Vocab size:", vocab_size)

    # Compute the maximum target length from your dataset.
    # We disable adding special tokens here so we can manually account for them.
    df_texts = df['text'].tolist()
    max_tgt_length = max(
        len(tokenizer.encode(t, add_special_tokens=False)) for t in df_texts) + 2  # +2 for <BOS> and <EOS>
    print("Max target length:", max_tgt_length)

    # Retrieve the special token IDs
    BOS_ID = tokenizer.bos_token_id
    PAD_ID = tokenizer.pad_token_id
    PAD_IDX = PAD_ID
    EOS_ID = tokenizer.eos_token_id
    UNK_ID = tokenizer.unk_token_id
    print("Special token IDs:", BOS_ID, PAD_ID, EOS_ID, UNK_ID)

    # Initialize your data processor with the GPT‑2 tokenizer.
    # (Assuming your ImageTxtProcessor class can work with a Transformers tokenizer.)
    data_processor = ImageTxtProcessor(tokenizer, args.p_aug, args.image_padding, args.img_width, True, True)

elif args.bpe and not args.gpt2_tokenizer :
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    df_texts = df['text'].tolist()
    tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
    trainer = BpeTrainer(
        vocab_size=args.bpe_size,  # choose a size that suits your data
        special_tokens=["<BOS>", "<PAD>", "<EOS>", "<UNK>"]
    )
    tokenizer.normalizer = Sequence([NFD()])  # Normalization: optional
    tokenizer.pre_tokenizer = Whitespace()  # Pre-tokenizer: splits on whitespace

    # Train on your texts
    tokenizer.train_from_iterator(df_texts, trainer=trainer)
    # Optionally, you could save the trained tokenizer
    # tokenizer.save("bpe_tokenizer.json")

    # Or if you already have a pre-trained tokenizer,
    # you can do: tokenizer = Tokenizer.from_file("bpe_tokenizer.json")

    vocab_size = tokenizer.get_vocab_size()
    print("Vocab size:", vocab_size)
    # You can keep the same approach, or set another strategy
    max_tgt_length = max(len(tokenizer.encode(t).ids) for t in df_texts) + 2

    # Example of retrieving special token IDs from the tokenizer
    BOS_ID = tokenizer.token_to_id("<BOS>")
    PAD_ID = tokenizer.token_to_id("<PAD>")
    EOS_ID = tokenizer.token_to_id("<EOS>")
    UNK_ID = tokenizer.token_to_id("<UNK>")
    PAD_IDX = PAD_ID
    print(BOS_ID, PAD_ID, EOS_ID, UNK_ID)
    data_processor = ImageTxtProcessor(tokenizer, args.p_aug, args.image_padding, args.img_width, True)
else :
    max_tgt_length = df['text'].str.len().max() + 2
    UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 3, 1, 0, 2
    special_tokens = {'<BOS>': 0, '<PAD>': 1, '<EOS>': 2, '<UNK>': 3}
    unique_chars = sorted(set(''.join(df['text'])))
    char_map = {char: idx + 4 for idx, char in enumerate(unique_chars)}
    char_map = {**special_tokens, **char_map}
    vocab_size = len(char_map)
    print(char_map)
    data_processor = ImageTxtProcessor(char_map, args.p_aug, args.image_padding, args.img_width)

if rank == 0:
    print("max_tgt_length : ", max_tgt_length)

train_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)
if valid_df is not None:
    valid_df.reset_index(drop=True, inplace=True)

if rank == 0:
    print("train_df len : ", len(train_df))
    if valid_df is not None:
        print("valid_df len : ", len(valid_df))
    print("test_df len : ", len(test_df))


if args.bpe and not args.gpt2_tokenizer :
    # Count tokens in test_df['text']
    total_tokens = sum(len(tokenizer.encode(text).ids) for text in test_df['text'])
    print("Total tokens in test set:", total_tokens)

if args.train_with_synthesized_lmdb :
    if args.DDP and world_size > 1:
        lmdb_dir = args.lmdb_dir
        lmdb_files = sorted(glob.glob(f"{lmdb_dir}/*.lmdb"))
        print("lmdb_files : ", lmdb_files)
        num_files = len(lmdb_files)
        print("len of lmdb : ", num_files)

        processes_per_file = world_size // num_files
        if processes_per_file == 0:
            file_index = rank
        else:
            file_index = rank // processes_per_file

        assigned_file = lmdb_files[file_index]

        print(f"Rank {rank} using LMDB file: {assigned_file}")

        # 1. Create the dataset as usual
        train_dataset = LMDBDataset(
            lmdb_path=assigned_file,
            processor=data_processor,
            max_tgt_length=max_tgt_length,
            height=args.img_height,
            width=args.img_width,
            do_aug=do_aug
        )

        # =========================================================================
        # ### CRITICAL FIX: SYNCHRONIZE DATASET LENGTHS ###
        # We are already inside the DDP block, so we just run the sync logic here.
        # =========================================================================

        # A. Wrap the length in a tensor on the GPU so NCCL can talk to it
        local_len = torch.tensor([len(train_dataset)], dtype=torch.long, device=device)

        # B. Find the MINIMUM length across all 8 GPUs
        dist.all_reduce(local_len, op=dist.ReduceOp.MIN)
        min_len = local_len.item()

        # C. Print info (only Rank 0 needs to talk)
        if rank == 0:
            print(f"Aligning dataset lengths for DDP stability.")
            print(f"Original lengths varied. Truncating all ranks to min length: {min_len}")

        # D. Force the dataset to be exactly 'min_len' long
        # This chops off the extra 500 samples from Ranks 4-7 so they stop at the exact same time as Ranks 0-3.
        train_dataset = torch.utils.data.Subset(train_dataset, range(min_len))
    else :
        train_dataset = LMDBDataset(lmdb_path=args.lmdb_path,
            processor=data_processor,
            max_tgt_length=max_tgt_length, height=args.img_height, width=args.img_width, do_aug=do_aug)
else :
    train_dataset = IAMDataset(root_dir=os.path.join(args.train_data_dir, 'image/'),
                               df=train_df,
                               processor=data_processor,
                               max_tgt_length=max_tgt_length, height=args.img_height, width=args.img_width, do_aug=do_aug)

# --- Valid/Test datasets ---
if do_train_valid_test and (valid_df is not None):
    valid_dataset = IAMDataset(
        root_dir=os.path.join(args.train_data_dir, 'image/'),
        df=valid_df,
        processor=data_processor,
        max_tgt_length=max_tgt_length,
        height=args.img_height,
        width=args.img_width,
        do_aug=False
    )
else:
    valid_dataset = None

test_dataset = IAMDataset(
    root_dir=os.path.join(args.train_data_dir, 'image/'),
    df=test_df,
    processor=data_processor,
    max_tgt_length=max_tgt_length,
    height=args.img_height,
    width=args.img_width,
    do_aug=False
)

eval_dataset = valid_dataset if valid_dataset is not None else test_dataset

print(f"[Rank {rank}] training examples : {len(train_dataset)}")
print(f"[Rank {rank}] eval (valid-or-test) examples : {len(eval_dataset)}")
print(f"[Rank {rank}] test examples : {len(test_dataset)}")

if args.DDP and world_size > 1:
    if args.train_with_synthesized_lmdb:
        # --- LMDB MODE (Safe to drop data) ---
        local_replicas = processes_per_file
        train_sampler = DistributedSampler(train_dataset, num_replicas=local_replicas, rank=rank % processes_per_file,
                                           shuffle=True, drop_last=True)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler,
                                      num_workers=num_cpus, pin_memory=True, drop_last=True, persistent_workers=True)

    else:
        # --- IAM MODE (Do NOT drop data) ---
        # FIX 1: Set shuffle=True for training!
        # FIX 2: drop_last=False (Default) enables padding so no data is lost and no hangs occur.
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank,
                                           shuffle=True, drop_last=False)

        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler,
                                      num_workers=num_cpus, pin_memory=True,
                                      drop_last=False, persistent_workers=True)

    # --- Debug Print ---
    # Since IAM mode uses padding (drop_last=False), these numbers will match across ranks.
    print(f"[Rank {rank}] Train DataLoader length: {len(train_dataloader)} batches")

    eval_sampler = DistributedSampler(eval_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    eval_dataloader = DataLoader(
        eval_dataset, batch_size=args.batch_size, sampler=eval_sampler,
        num_workers=num_cpus, pin_memory=True, persistent_workers=True
    )

    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    if do_train_valid_test:
        test_dataloader = DataLoader(
            test_dataset, batch_size=args.batch_size, sampler=test_sampler,
            num_workers=num_cpus, pin_memory=True, persistent_workers=True
        )

    log_memory_usage_ddp("before load the model", device)
else:
    # Single GPU/CPU Case
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=num_cpus, drop_last=False, persistent_workers=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size,
                                 num_workers=num_cpus, persistent_workers=True)
    if do_train_valid_test:
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                     num_workers=num_cpus, persistent_workers=True)

    log_memory_usage("before load the model", device)


model = DRetHTR(decoder_mode=args.decoder, vocab_size=vocab_size,
                    image_size=(args.img_height, args.img_width), patch_size=args.patch_size,
                    num_channels=args.num_channels, img_emb_dropout = args.img_emb_dropout,patch_order=args.patch_order,\
                    embed_dim=args.embed_dim,d_model = args.d_model,decoder_attention_heads=args.decoder_attention_heads, decoder_ffn_dim=args.decoder_ffn_dim,
                    decoder_depth=args.decoder_depth, decoder_dropout=args.decoder_dropout, decoder_emb_dropout=args.decoder_emb_dropout,
                    cnn_dropout = args.cnn_dropout,
                    feature_extractor= args.feature_extractor,
                    D_norm=args.D_norm, ret_norm=args.ret_norm, gamma_subtracter=args.gamma_subtracter, various_gamma_in_heads=args.various_gamma_in_heads, increase_gamma_along_layers=args.increase_gamma_along_layers,
                text_length=max_tgt_length).to(device)

if args.DDP :
    log_memory_usage_ddp("before load the model", device)
else :
    log_memory_usage('after load the model', device)

def init_weights(model, exclude_modules):
    for module in model.modules():
        if module in exclude_modules:
            continue
        if isinstance(module, nn.Linear):
            if args.weight_init == "sd0.02":
                torch.nn.init.normal_(module.weight, mean=0, std=0.02)
            elif args.weight_init == "He":
                torch.nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                module.bias.data.fill_(args.bias_init)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)
        else:
            for name, param in module.named_parameters():
                if 'weight' in name and param.dim() > 1:
                    if args.weight_init == "sd0.02":
                        torch.nn.init.normal_(param, mean=0, std=0.02)
                    elif args.weight_init == "He":
                        torch.nn.init.kaiming_uniform_(param, nonlinearity='relu')
                elif 'bias' in name:
                    param.data.fill_(args.bias_init)

if args.weight_init != 'None':
    if args.use_pre_trained_backbone :
        exclude_modules = list(model.feature_extractor.features) if "ResNet" in args.feature_extractor or "efficientnet" in args.feature_extractor or "regnet" in args.feature_extractor else []
    else :
        exclude_modules =[]
    init_weights(model, exclude_modules)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

total_params = count_parameters(model)
print(f"Total number of parameters: {total_params}")
base_path = args.base_path
os.makedirs(base_path, exist_ok=True)  # Create directories and ignore if they exist
if not os.path.exists(base_path):
    base_path = "./"  # Set to current directory if the base path doesn't exist
model_dir = os.path.join(base_path, args.modeldir)
run_dir = os.path.join(model_dir, args.run_name)
os.makedirs(run_dir, exist_ok=True)
weight_path = os.path.join(run_dir, "ckpt.pt")

from collections import OrderedDict

# Load previous model if exists
if args.DDP and world_size > 1 :
    if not args.load_weight_from_non_DDP :
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    broadcast_buffers=False,   # ← stops BatchNorm stats from causing the same hang
                    find_unused_parameters = False  # <-- DIAGNOSTIC
        )

    try:
        ckpt = torch.load(weight_path)
        model.load_state_dict(ckpt)
        print("Successfully loaded previous model weights")
    except Exception as e:
        print("Failed to load previous model weights:", e)

    if args.load_weight_from_non_DDP :
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    broadcast_buffers=False  # ← stops BatchNorm stats from causing the same hang
                    )
else :
    try:
        if device == "cpu":
            ckpt = torch.load(weight_path, map_location=torch.device('cpu'))
        else:
            # New addition: if loading DDP-trained weights for non-DDP inference, remove the "module." prefix
            if args.load_weight_from_DDP:
                ckpt = torch.load(weight_path, map_location=lambda storage, loc: storage.cuda(0))
                ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
            else :
                ckpt = torch.load(weight_path, map_location=lambda storage, loc: storage.cuda(0))

        if args.skip_emb_generator_weights :
            # Remove incompatible vocabulary-dependent weights
            for key in ["embed_tokens.weight", "generator.weight"]:
                if key in ckpt:
                    print(f"Skipping incompatible key: {key}")
                    del ckpt[key]

        model.load_state_dict(ckpt, strict=False)
        print("Successfully loaded previous model weights")
    except Exception as e:
        print("Failed to load previous model weights:", e)



if args.label_smooth:
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.4)
else:
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

# Setup optimizer and scheduler
if args.lr_scheduler == "StepLR":
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.9213761685145376)
elif args.lr_scheduler == "CosineAnnealingLR":
    steps_per_epoch = len(train_dataloader)
    total_steps = args.cosineRestartEpoch * steps_per_epoch
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
else:
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# Training loop
# if args.train_with_synthesized_lmdb :
#     NUM_EPOCHS = 1
# else :
NUM_EPOCHS = args.epochs
train_losses = []
output_dir = os.path.join(args.modeldir, args.run_name)
file_path = os.path.join(output_dir, "loss.txt")

s_e = get_last_recorded_epoch(file_path) + 1
print("last epoch was ",s_e-1,"start epoch is ", s_e)
if s_e == 1 :
    prev_val_loss = 999 ; prev_val_cer=999
else :
    if args.mode != 'visualize' and "test" not in args.mode:
        prev_val_loss, prev_val_cer = evaluate(model, eval_dataloader)
    else :
        prev_val_loss = 999; prev_val_cer = 999

print('Previous Validation Loss:', prev_val_loss)
print("args.mode : ",args.mode)



# -------------------- helpers --------------------
def ddp_mean(value: float, device):
    """All-reduce mean for a scalar float across all ranks."""
    if dist.is_initialized():
        t = torch.tensor([float(value)], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= dist.get_world_size()
        return float(t.item())
    return float(value)

def ddp_broadcast_from_rank0(value, device, fallback=-1.0):
    """
    Broadcast a scalar from rank0 to all ranks.
    If this rank has None, send fallback; rank0 sends real value.
    """
    if not dist.is_initialized():
        return value
    rank = dist.get_rank()
    if rank == 0:
        t = torch.tensor([float(value)], device=device)
    else:
        t = torch.tensor([float(fallback if value is None else value)], device=device)
    dist.broadcast(t, src=0)
    return float(t.item())

# ============================================================
# ===================== TRAINING MODES =======================
# ============================================================

eval_name = "Valid" if do_train_valid_test else "Test"

if args.mode == 'train_inference_recurrent':
    # Initial evaluation before entering the training loop (runs on all ranks)
    prev_test_cer = evaluate_decoding(
        model, eval_dataloader, data_processor, max_tgt_length, args, device,
        print_str=args.print_eval, beam_search=args.beam_during_test, mode="recurrent"
    )

    for epoch in range(s_e, NUM_EPOCHS + 1):
        if args.DDP and world_size > 1:
            train_sampler.set_epoch(epoch)
        else:
            rank = 0

        start_time = timer()

        # -------------------- Train one epoch --------------------
        train_loss, train_cer = train_epoch(
            model, optimizer, loss_fn, train_dataloader, eval_dataloader,
            data_processor, max_tgt_length, weight_path, args, device, rank
        )

        # Make train_loss global mean across ranks (so rank0 prints global loss)
        if args.DDP and dist.is_initialized():
            train_loss = ddp_mean(train_loss, device)

            # train_cer exists only on rank0; broadcast it so formatting never crashes
            # (even though we only print on rank0, this also keeps vars consistent)
            train_cer = ddp_broadcast_from_rank0(train_cer, device, fallback=-1.0)
        else:
            # single GPU: train_cer should exist
            train_cer = float(train_cer)

        # -------------------- Evaluate sometimes --------------------
        test_cer = prev_test_cer

        if epoch % args.eval_cycle_epoch == 0:
            test_cer = evaluate_decoding(
                model, eval_dataloader, data_processor, max_tgt_length, args, device,
                print_str=args.print_eval, beam_search=args.beam_during_test, mode="recurrent"
            )

            # Only rank 0 saves checkpoints
            if rank == 0:
                if test_cer < prev_test_cer or args.train_with_synthesized_lmdb:
                    if args.train_with_synthesized_lmdb:
                        part = get_next_checkpoint_number(run_dir)
                        weight_path_epoch = os.path.join(run_dir, f"ckpt_{part}.pt")
                        print(f"Next checkpoint path: {weight_path_epoch}", flush=True)
                        torch.save(model.state_dict(), weight_path_epoch)
                        torch.save(model.state_dict(), weight_path)
                    else:
                        torch.save(model.state_dict(), weight_path)
                    prev_test_cer = test_cer

                print(f"Epoch: {epoch}, Global {eval_name} CER: {test_cer:.4f}", flush=True)

            if args.DDP and dist.is_initialized():
                dist.barrier()

        end_time = timer()

        # -------------------- Print epoch summary (rank0 only) --------------------
        if (not args.DDP) or (rank == 0):
            # If you used fallback=-1.0 you can decide how to display it:
            if train_cer < 0:
                train_cer_str = "N/A"
            else:
                train_cer_str = f"{train_cer:.4f}"

            print(
                f"Epoch: {epoch}, Train Loss: {train_loss:.3f}, Train CER: {train_cer_str}, "
                f"{eval_name} CER: {test_cer:.4f}, Time: {end_time - start_time:.3f}, {args.run_name}",
                flush=True
            )

        train_losses.append(train_loss)

        scheduler.step()
        torch.cuda.empty_cache()
        gc.collect()

    if do_train_valid_test:
        final_test_cer = evaluate_decoding(
            model, test_dataloader, data_processor, max_tgt_length, args, device,
            print_str=args.print_eval, beam_search=args.beam_during_test, mode="recurrent"
        )
        if (not args.DDP) or (rank == 0):
            print(f"FINAL TEST CER: {final_test_cer:.4f} | {args.run_name}", flush=True)

        # keep ranks in sync before teardown
        if args.DDP and dist.is_initialized():
            dist.barrier()

    if args.DDP and world_size > 1:
        dist.destroy_process_group()

elif args.mode == 'train_inference_kv_cached':
    # Initial evaluation before entering the training loop (runs on all ranks)
    prev_test_cer = evaluate_decoding(
        model, eval_dataloader, data_processor, max_tgt_length, args, device,
        print_str=args.print_eval, beam_search=args.beam_during_test, mode="kv_cached"
    )

    for epoch in range(s_e, NUM_EPOCHS + 1):
        if args.DDP and world_size > 1:
            train_sampler.set_epoch(epoch)
        else:
            rank = 0

        start_time = timer()

        # -------------------- Train one epoch --------------------
        train_loss, train_cer = train_epoch(
            model, optimizer, loss_fn, train_dataloader, eval_dataloader,
            data_processor, max_tgt_length, weight_path, args, device, rank
        )

        # Make train_loss global mean across ranks (so rank0 prints global loss)
        if args.DDP and dist.is_initialized():
            train_loss = ddp_mean(train_loss, device)

            # train_cer exists only on rank0; broadcast it so formatting never crashes
            train_cer = ddp_broadcast_from_rank0(train_cer, device, fallback=-1.0)
        else:
            train_cer = float(train_cer)

        # -------------------- Evaluate sometimes --------------------
        test_cer = prev_test_cer

        if epoch % args.eval_cycle == 0:
            test_cer = evaluate_decoding(
                model, eval_dataloader, data_processor, max_tgt_length, args, device,
                print_str=args.print_eval, beam_search=args.beam_during_test, mode="kv_cached"
            )

            # Only rank 0 saves checkpoints
            if rank == 0:
                if test_cer < prev_test_cer or args.train_with_synthesized_lmdb:
                    if args.train_with_synthesized_lmdb:
                        part = get_next_checkpoint_number(run_dir)
                        weight_path_epoch = os.path.join(run_dir, f"ckpt_{part}.pt")
                        print(f"Next checkpoint path: {weight_path_epoch}", flush=True)
                        torch.save(model.state_dict(), weight_path_epoch)
                        torch.save(model.state_dict(), weight_path)
                    else:
                        torch.save(model.state_dict(), weight_path)
                    prev_test_cer = test_cer

                print(f"Epoch: {epoch}, Global {eval_name} CER: {test_cer:.4f}", flush=True)

            if args.DDP and dist.is_initialized():
                dist.barrier()

        end_time = timer()

        # -------------------- Print epoch summary (rank0 only) --------------------
        if (not args.DDP) or (rank == 0):
            if train_cer < 0:
                train_cer_str = "N/A"
            else:
                train_cer_str = f"{train_cer:.4f}"

            print(
                f"Epoch: {epoch}, Train Loss: {train_loss:.3f}, Train CER: {train_cer_str}, "
                f"{eval_name} CER: {test_cer:.4f}, Time: {end_time - start_time:.3f}, {args.run_name}",
                flush=True
            )

        train_losses.append(train_loss)

        scheduler.step()
        torch.cuda.empty_cache()
        gc.collect()

    if do_train_valid_test:
        final_test_cer = evaluate_decoding(
            model, test_dataloader, data_processor, max_tgt_length, args, device,
            print_str=args.print_eval, beam_search=args.beam_during_test, mode="recurrent"
        )
        if (not args.DDP) or (rank == 0):
            print(f"FINAL TEST CER: {final_test_cer:.4f} | {args.run_name}", flush=True)

        # keep ranks in sync before teardown
        if args.DDP and dist.is_initialized():
            dist.barrier()

    if args.DDP and world_size > 1:
        dist.destroy_process_group()

elif args.mode =='test_recurrent' :
    start_time = timer()
    test_cer = evaluate_decoding(model, eval_dataloader, data_processor, max_tgt_length, args, device, print_str=args.print_eval, beam_search=False, mode="recurrent")
    end_time = timer()
    print(f"test cer : {test_cer:.4f} | {args.run_name} | time : {end_time-start_time:.3f}")

elif args.mode =='test_recurrent_beam' :
    start_time = timer()
    test_cer = evaluate_decoding(model, eval_dataloader, data_processor, max_tgt_length, args, device, print_str=args.print_eval, beam_search=True, mode="recurrent")
    end_time = timer()
    print(f"test cer : {test_cer:.4f} | {args.run_name} | time : {end_time-start_time:.3f}")

elif args.mode =='test_recurrent_beam_m' :
    start_time = timer()
    test_cer = evaluate_decoding_memory_check(model, eval_dataloader, data_processor, max_tgt_length, args, device, print_str=args.print_eval, beam_search=True, mode="recurrent")
    end_time = timer()
    print(f"test cer : {test_cer:.4f} | {args.run_name} | time : {end_time-start_time:.4f}")

elif args.mode =='test_recurrent_wer' :
    start_time = timer()
    test_cer, test_wer = evaluate_decoding(model, eval_dataloader, data_processor, max_tgt_length, args, device, print_str=args.print_eval, beam_search=False, mode="recurrent", wer_flag=True)
    end_time = timer()
    print(f"test cer : {test_cer:.4f} | test wer : {test_wer:.4f} | {args.run_name} | time : {end_time-start_time:.3f}")

elif args.mode =='test_recurrent_beam_wer' :
    start_time = timer()
    test_cer, test_wer = evaluate_decoding(model, eval_dataloader, data_processor, max_tgt_length, args, device, print_str=args.print_eval, beam_search=True, mode="recurrent", wer_flag=True)
    end_time = timer()
    print(f"test cer : {test_cer:.4f} | test wer : {test_wer:.4f} | {args.run_name} | time : {end_time-start_time:.3f}")

elif args.mode =='test_kv_cached' :
    start_time = timer()
    test_cer = evaluate_decoding(model, eval_dataloader, data_processor, max_tgt_length, args, device, print_str=args.print_eval, beam_search=False, mode="kv_cached")
    end_time = timer()
    print(f"test cer : {test_cer:.4f} | {args.run_name} | time : {end_time-start_time:.3f}")

elif args.mode =='test_kv_cached_beam' :
    start_time = timer()
    test_cer = evaluate_decoding(model, eval_dataloader, data_processor, max_tgt_length, args, device, print_str=args.print_eval, beam_search=True, mode="kv_cached")
    end_time = timer()
    print(f"test cer : {test_cer:.4f} | {args.run_name} | time : {end_time-start_time:.3f}")

elif args.mode =='test_kv_cached_beam_m' :
    start_time = timer()
    test_cer = evaluate_decoding_memory_check(model, eval_dataloader, data_processor, max_tgt_length, args, device, print_str=args.print_eval, beam_search=True, mode="kv_cached")
    end_time = timer()
    print(f"test cer : {test_cer:.4f} | {args.run_name} | time : {end_time-start_time:.3f}")

elif args.mode =='test_kv_cached_wer' :
    start_time = timer()
    test_cer, test_wer = evaluate_decoding(model, eval_dataloader, max_tgt_length, args, device, print_str=args.print_eval, beam_search=False, mode="kv_cached", wer_flag=True)
    end_time = timer()
    print(f"test cer : {test_cer:.4f} | test wer : {test_wer:.4f} | {args.run_name} | time : {end_time-start_time:.3f}")

elif args.mode =='test_kv_cached_beam_wer' :
    start_time = timer()
    test_cer, test_wer = evaluate_decoding(model, eval_dataloader, data_processor, max_tgt_length, args, device, print_str=args.print_eval, beam_search=True, mode="kv_cached", wer_flag=True)
    end_time = timer()
    print(f"test cer : {test_cer:.4f} | test wer : {test_wer:.4f} | {args.run_name} | time : {end_time-start_time:.3f}")