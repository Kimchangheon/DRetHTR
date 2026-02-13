import torch
from .decode import *

from datasets import load_metric
import os
import signal
import torch
import torch.distributed as dist
from timeit import default_timer as timer

UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 3, 1, 0, 2

cer_metric = load_metric("cer")
wer_metric = load_metric("wer")

def set_to_one_after_eos(tensor):
    mask = torch.cumsum(tensor == EOS_IDX, dim=1) > 0
    tensor[mask] = PAD_IDX
    return tensor

def compute_cer(data_processor, pred_ids, label_ids, print_str=False):
    pred_ids = set_to_one_after_eos(pred_ids)
    pred_str = data_processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_ids[label_ids == -100] = PAD_IDX
    label_str = data_processor.batch_decode(label_ids, skip_special_tokens=True)
    # After converting label IDs to strings
    for idx, ref in enumerate(label_str):
        if not ref.strip():
            print(f"Empty reference at index {idx}")
    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    if print_str :
        for i in range(len(pred_str)) :
            print("pred_str  : ", pred_str[i], "label_str : ", label_str[i], "CER : ",cer)
    return cer

def compute_wer(data_processor, pred_ids, label_ids, print_str=False):
    pred_ids = set_to_one_after_eos(pred_ids)
    pred_str = data_processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_ids[label_ids == -100] = PAD_IDX
    label_str = data_processor.batch_decode(label_ids, skip_special_tokens=True)

    # Print references with issues
    for idx, ref in enumerate(label_str):
        if not ref.strip():
            print(f"Empty reference at index {idx}")

    # Compute WER
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    if print_str:
        for i in range(len(pred_str)):
            print("Prediction  : ", pred_str[i])
            print("Reference   : ", label_str[i])

    return wer




def _ddp_abort_all(msg: str, rank: int, device=None):
    """
    Best-effort: make *every* rank exit if one rank hits an exception.
    - Tries dist.abort() (fastest).
    - Falls back to SIGTERM current process.
    """
    print(f"[Rank {rank}] FATAL: {msg}", flush=True)
    try:
        if dist.is_available() and dist.is_initialized():
            # abort() exists in newer torch; if not, except will catch
            dist.abort()
            return
    except Exception:
        pass

    # Fallback: kill this process. In SLURM, job usually gets torn down quickly.
    try:
        os.kill(os.getpid(), signal.SIGTERM)
    except Exception:
        raise RuntimeError(msg)

def train_epoch(model, optimizer, loss_fn,
                train_dataloader, eval_dataloader,
                data_processor, max_tgt_length,
                weight_path, args, device, rank):
    """
    - Computes/returns TRAIN CER only on rank 0 (computed on CPU).
    - Wraps the whole iteration to avoid "one-rank exception -> others hang".
    - On any exception, aborts all ranks immediately.
    """
    ddp_on = bool(getattr(args, "DDP", False)) and dist.is_available() and dist.is_initialized()
    mode = "recurrent" if "RetNet" in args.decoder else "kv_cached"

    # --- optional initial eval in LMDB mode ---
    try:
        if args.train_with_synthesized_lmdb:
            prev_test_cer = evaluate_decoding(
                model, eval_dataloader, data_processor, max_tgt_length,
                args, device, beam_search=False, mode=mode
            )
        else:
            prev_test_cer = 999
    except Exception as e:
        _ddp_abort_all(f"Exception during initial evaluate_decoding: {repr(e)}", rank)
        raise  # for non-DDP / safety

    model.train()
    total_loss = 0.0

    # rank-0-only train CER accumulation
    total_cer_rank0 = 0.0
    cer_batches_rank0 = 0

    batch_counter = 0
    start_time = timer()

    try:
        for batch in train_dataloader:
            batch_counter += 1

            optimizer.zero_grad(set_to_none=True)

            # Move to GPU
            for k, v in batch.items():
                batch[k] = v.to(device, non_blocking=True)

            src = batch["pixel_values"]
            tgt = batch["labels"]
            tgt = tgt.clone()  # avoid in-place side effects if tensors are reused
            tgt.masked_fill_(tgt == -100, PAD_IDX)

            decoder_input_ids = tgt[:, :-1]

            pad_positions = None
            img_pad_ratios = None
            if args.feed_pad_positions:
                pad_positions = (decoder_input_ids == PAD_IDX).int().argmax(dim=1)
            if args.feed_img_pad_ratios:
                img_pad_ratios = batch["ratio"]

            logits = model(src, decoder_input_ids, pad_positions, img_pad_ratios)

            tgt_out = tgt[:, 1:]
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))

            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())

            # ---- TRAIN CER: rank 0 only; compute on CPU to avoid GPU sync stalls ----
            if rank == 0:
                with torch.no_grad():
                    tgt_tokens = torch.argmax(logits, dim=2)
                    cer = compute_cer(
                        data_processor=data_processor,
                        pred_ids=tgt_tokens.detach().cpu(),
                        label_ids=tgt_out.detach().cpu()
                    )
                total_cer_rank0 += float(cer)
                cer_batches_rank0 += 1

            # ---- periodic LMDB eval/ckpt ----
            if args.train_with_synthesized_lmdb and (batch_counter % 1000 == 0 or batch_counter == len(train_dataloader)):
                end_time = timer()
                end_time_decode = end_time
                end_time_decode_save = end_time

                if (batch_counter % (1000 * args.eval_cycle) == 0) or (batch_counter == len(train_dataloader)):
                    test_cer = evaluate_decoding(
                        model, eval_dataloader, data_processor, max_tgt_length,
                        args, device, beam_search=False, mode=mode
                    )
                    end_time_decode = timer()

                    if rank == 0 and prev_test_cer >= test_cer:
                        torch.save(model.state_dict(), weight_path)
                        end_time_decode_save = timer()
                    else:
                        end_time_decode_save = end_time_decode  # Set same as decode time when not saving

                    # IMPORTANT: barrier must be reached by ALL ranks if used
                    if ddp_on:
                        dist.barrier()
                else:
                    test_cer = prev_test_cer

                # Only rank 0 prints
                if rank == 0:
                    avg_train_cer = total_cer_rank0 / max(cer_batches_rank0, 1)
                    print(
                        f"[Rank 0] After {batch_counter} batches: "
                        f"Train Loss(last)={loss.item():.4f}, "
                        f"Train CER(avg)={avg_train_cer:.4f}, "
                        f"Global Test CER={test_cer:.4f}, "
                        f"Train Time={end_time - start_time:.3f} "
                        f"Decode Time={end_time_decode - end_time:.3f} "
                        f"Save Time={end_time_decode_save - end_time_decode:.3f}",
                        flush=True
                    )

                prev_test_cer = test_cer
                model.train()
                start_time = timer()

    except Exception as e:
        # If ANY rank hits an exception, immediately abort all ranks to avoid a hang on next allreduce.
        _ddp_abort_all(
            f"Exception inside train loop at batch {batch_counter}/{len(train_dataloader)}: {repr(e)}",
            rank
        )
        raise  # for visibility in single-rank / logs

    # ---- return values ----
    avg_loss = total_loss / max(len(train_dataloader), 1)
    if rank == 0:
        avg_cer = total_cer_rank0 / max(cer_batches_rank0, 1)
        return avg_loss, avg_cer
    else:
        return avg_loss, None
def evaluate(model, loss_fn, eval_dataloader, data_processor, args, device):
    model.eval()
    total_loss = 0
    total_cer = 0.0
    with torch.no_grad():
        for batch in eval_dataloader:
            for k, v in batch.items():
                batch[k] = v.to(device)

            src = batch['pixel_values']
            tgt = batch['labels']
            tgt.masked_fill_(tgt == -100, PAD_IDX)
            decoder_input_ids = tgt[:, :-1]

            pad_positions = None;
            img_pad_ratios = None
            if args.feed_pad_positions:
                pad_positions = (decoder_input_ids == PAD_IDX).int().argmax(dim=1)
            if args.feed_img_pad_ratios:
                img_pad_ratios = batch['ratio']

            logits = model(src, decoder_input_ids, pad_positions, img_pad_ratios)

            tgt_out = tgt[:, 1:]
            tgt_tokens = torch.max(logits, dim=2)[1]

            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            total_loss += loss.item()
            cer = compute_cer(data_processor=data_processor, pred_ids=tgt_tokens, label_ids=tgt_out)
            total_cer += cer

    return total_loss / len(eval_dataloader), total_cer / len(eval_dataloader)

def evaluate_decoding(model, eval_dataloader, data_processor, max_tgt_length, args, device, print_str=False, beam_search=False, mode="recurrent", wer_flag=False):
    model.eval()
    total_cer = 0.0
    total_wer = 0.0
    total_batches = 0  # to track how many batches we processed
    error_flag = 0  # 0 = OK, 1 = some error happened

    try :
        with torch.no_grad():
            for batch in eval_dataloader:
                total_batches += 1
                for k, v in batch.items():
                    batch[k] = v.to(device)

                src = batch['pixel_values']
                tgt = batch['labels']
                tgt.masked_fill_(tgt == -100, PAD_IDX)

                tgt_out = tgt[:, 1:]

                img_pad_ratios = None
                if args.feed_img_pad_ratios:
                    img_pad_ratios = batch['ratio']

                if beam_search :
                    if mode == "recurrent" :
                        if args.retnorm_inference :
                            tgt_tokens = beam_decode_recurrent_retnorm(model, src, img_pad_ratios=img_pad_ratios,
                                                               max_len=max_tgt_length, start_symbol=BOS_IDX,
                                                               end_symbol=EOS_IDX,
                                                               pad_idx=PAD_IDX, beam_width=args.beam_width, device=device)
                        else :
                            tgt_tokens = beam_decode_recurrent(model, src, img_pad_ratios=img_pad_ratios,
                                                                  max_len=max_tgt_length, start_symbol=BOS_IDX,
                                                                  end_symbol=EOS_IDX,
                                                                  pad_idx=PAD_IDX, beam_width=args.beam_width, device=device)
                    elif mode == "kv_cached" :
                        tgt_tokens = beam_decode_kv_cached(model, src, img_pad_ratios=img_pad_ratios,
                                                              max_len=max_tgt_length, start_symbol=BOS_IDX,
                                                              end_symbol=EOS_IDX,
                                                              pad_idx=PAD_IDX, beam_width=args.beam_width, device=device)
                    elif mode =="vanilla" :
                        tgt_tokens = beam_decode_vanilla(model, src, img_pad_ratios=img_pad_ratios,
                                                              max_len=max_tgt_length, start_symbol=BOS_IDX,
                                                              end_symbol=EOS_IDX,
                                                              pad_idx=PAD_IDX, beam_width=args.beam_width, device=device)

                else :
                    if mode == "recurrent" :
                        if args.retnorm_inference :
                            tgt_tokens = greedy_decode_recurrent_retnorm(model, src, img_pad_ratios=img_pad_ratios,
                                                                 max_len=max_tgt_length, start_symbol=BOS_IDX,
                                                                 end_symbol=EOS_IDX,
                                                                 pad_idx=PAD_IDX, device=device)
                        else :
                            tgt_tokens = greedy_decode_recurrent(model, src, img_pad_ratios=img_pad_ratios,
                                                                  max_len=max_tgt_length, start_symbol=BOS_IDX,
                                                                  end_symbol=EOS_IDX,
                                                                  pad_idx=PAD_IDX, device=device)
                    elif mode == "kv_cached" :
                        tgt_tokens = greedy_decode_kv_cached(model, src, img_pad_ratios=img_pad_ratios,
                                                              max_len=max_tgt_length, start_symbol=BOS_IDX,
                                                              end_symbol=EOS_IDX,
                                                              pad_idx=PAD_IDX, device=device)
                    elif mode =="vanilla" :
                        tgt_tokens = greedy_decode_vanilla(model, src, img_pad_ratios=img_pad_ratios,
                                                              max_len=max_tgt_length, start_symbol=BOS_IDX,
                                                              end_symbol=EOS_IDX,
                                                              pad_idx=PAD_IDX, device=device)

                cer = compute_cer(data_processor=data_processor, pred_ids=tgt_tokens, label_ids=tgt_out, print_str=print_str)
                total_cer += cer
                if wer_flag :
                    wer = compute_wer(data_processor=data_processor, pred_ids=tgt_tokens, label_ids=tgt_out, print_str=print_str)
                    total_wer += wer
    except Exception as e:
        # mark that something went wrong on this rank
        print(f"[Rank {dist.get_rank() if dist.is_initialized() else 0}] "
              f"ERROR in evaluate_decoding: {e}", flush=True)
        error_flag = 1


    # ---- global aggregation (runs on ALL ranks, even if some had errors) ----
    if args.DDP and dist.is_initialized():
        cer_tensor   = torch.tensor(float(total_cer),   device=device)
        batch_tensor = torch.tensor(float(total_batches), device=device)
        err_tensor   = torch.tensor(error_flag,         device=device)

        dist.all_reduce(cer_tensor,   op=dist.ReduceOp.SUM)
        dist.all_reduce(batch_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(err_tensor,   op=dist.ReduceOp.SUM)

        global_cer_sum    = cer_tensor.item()
        global_batch_sum  = batch_tensor.item()
        any_error_happened = err_tensor.item() > 0

        # Avoid division by zero
        if global_batch_sum > 0:
            global_cer = global_cer_sum / global_batch_sum
        else:
            # If absolutely nothing was processed anywhere, return "worst" CER
            global_cer = 1.0

        global_wer = None
        if wer_flag:
            wer_tensor = torch.tensor(float(total_wer), device=device)
            dist.all_reduce(wer_tensor, op=dist.ReduceOp.SUM)
            global_wer_sum = wer_tensor.item()
            if global_batch_sum > 0:
                global_wer = global_wer_sum / global_batch_sum
            else:
                global_wer = 1.0

        # Just warn once (rank 0) if something went wrong, but DO NOT raise.
        if any_error_happened and dist.get_rank() == 0:
            print("[evaluate_decoding] WARNING: some ranks failed during evaluation; "
                  "returning partial / degraded CER (and WER).", flush=True)

        if wer_flag:
            return global_cer, global_wer
        else:
            return global_cer

    else:
        # non-DDP case
        if error_flag:
            raise RuntimeError("evaluate_decoding failed (single-process).")
        if wer_flag:
            return total_cer / max(total_batches, 1), total_wer / max(total_batches, 1)
        else:
            return total_cer / max(total_batches, 1)

def log_memory_usage(stage, device):
    allocated = torch.cuda.memory_allocated(device) / 1024 ** 2  # in MiB
    reserved = torch.cuda.memory_reserved(device) / 1024 ** 2  # in MiB
    peak_allocated = torch.cuda.max_memory_allocated(device) / 1024 ** 2  # Peak allocated memory in MiB
    peak_reserved = torch.cuda.max_memory_reserved(device) / 1024 ** 2  # Peak reserved memory in MiB
    print(f"{stage} - Memory Allocated: {allocated:.2f} MiB, Memory Reserved: {reserved:.2f} MiB, "
          f"Peak Allocated: {peak_allocated:.2f} MiB, Peak Reserved: {peak_reserved:.2f} MiB")
    return peak_allocated


def evaluate_decoding_memory_check(model, eval_dataloader, data_processor, max_tgt_length, args, device, print_str=False, beam_search=False, mode="recurrent", wer_flag=False):
    model.eval()
    total_cer = 0.0
    total_wer = 0.0
    total_batches = 0  # to track how many batches we processed
    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_dataloader):
            # Reset peak memory stats at the start of each batch
            torch.cuda.reset_peak_memory_stats(device)

            # Log memory before processing the batch
            print(f"Batch {batch_idx + 1}/{len(eval_dataloader)} - Before processing:")
            peak_before_processing = log_memory_usage("Before processing", device)
            total_batches += 1
            for k, v in batch.items():
                batch[k] = v.to(device)

            src = batch['pixel_values']
            tgt = batch['labels']
            tgt.masked_fill_(tgt == -100, PAD_IDX)

            tgt_out = tgt[:, 1:]

            img_pad_ratios = None
            if args.feed_img_pad_ratios:
                img_pad_ratios = batch['ratio']

            if beam_search :
                if mode == "recurrent" :
                    if args.retnorm_inference :
                        tgt_tokens = beam_decode_recurrent_retnorm(model, src, img_pad_ratios=img_pad_ratios,
                                                           max_len=max_tgt_length, start_symbol=BOS_IDX,
                                                           end_symbol=EOS_IDX,
                                                           pad_idx=PAD_IDX, beam_width=args.beam_width, device=device)
                    else :
                        tgt_tokens = beam_decode_recurrent(model, src, img_pad_ratios=img_pad_ratios,
                                                              max_len=max_tgt_length, start_symbol=BOS_IDX,
                                                              end_symbol=EOS_IDX,
                                                              pad_idx=PAD_IDX, beam_width=args.beam_width, device=device)
                elif mode == "kv_cached" :
                    tgt_tokens = beam_decode_kv_cached(model, src, img_pad_ratios=img_pad_ratios,
                                                          max_len=max_tgt_length, start_symbol=BOS_IDX,
                                                          end_symbol=EOS_IDX,
                                                          pad_idx=PAD_IDX, beam_width=args.beam_width, device=device)
                elif mode =="vanilla" :
                    tgt_tokens = beam_decode_vanilla(model, src, img_pad_ratios=img_pad_ratios,
                                                          max_len=max_tgt_length, start_symbol=BOS_IDX,
                                                          end_symbol=EOS_IDX,
                                                          pad_idx=PAD_IDX, beam_width=args.beam_width, device=device)

            else :
                if mode == "recurrent" :
                    if args.retnorm_inference :
                        tgt_tokens = greedy_decode_recurrent_retnorm(model, src, img_pad_ratios=img_pad_ratios,
                                                             max_len=max_tgt_length, start_symbol=BOS_IDX,
                                                             end_symbol=EOS_IDX,
                                                             pad_idx=PAD_IDX, device=device)
                    else :
                        tgt_tokens = greedy_decode_recurrent(model, src, img_pad_ratios=img_pad_ratios,
                                                              max_len=max_tgt_length, start_symbol=BOS_IDX,
                                                              end_symbol=EOS_IDX,
                                                              pad_idx=PAD_IDX, device=device)
                elif mode == "kv_cached" :
                    tgt_tokens = greedy_decode_kv_cached(model, src, img_pad_ratios=img_pad_ratios,
                                                          max_len=max_tgt_length, start_symbol=BOS_IDX,
                                                          end_symbol=EOS_IDX,
                                                          pad_idx=PAD_IDX, device=device)
                # elif mode =="vanilla" :
                    tgt_tokens = greedy_decode_vanilla(model, src, img_pad_ratios=img_pad_ratios,
                                                          max_len=max_tgt_length, start_symbol=BOS_IDX,
                                                          end_symbol=EOS_IDX,
                                                          pad_idx=PAD_IDX, device=device)

            # Log memory after decoding
            print(f"Batch {batch_idx + 1} - After decoding:")
            peak_after_decoding = log_memory_usage("After decoding", device)

            # Calculate peak memory difference due to decoding
            decoding_peak_diff = peak_after_decoding - peak_before_processing
            print(f"Batch {batch_idx + 1} - Peak Allocated Memory Difference (Decoding): {decoding_peak_diff:.2f} MiB")
            print()

            cer = compute_cer(data_processor=data_processor, pred_ids=tgt_tokens, label_ids=tgt_out, print_str=print_str)
            total_cer += cer
            if wer_flag :
                wer = compute_wer(data_processor=data_processor, pred_ids=tgt_tokens, label_ids=tgt_out, print_str=print_str)
                total_wer += wer
            # Cleanup after decoding
            del tgt_tokens
            torch.cuda.empty_cache()  # Free cached memory for the next batch
        # If using DDP, aggregate results from all processes
        if args.DDP and torch.distributed.is_initialized():
            cer_tensor = torch.tensor(total_cer, device=device)
            batch_tensor = torch.tensor(total_batches, device=device)
            torch.distributed.all_reduce(cer_tensor, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(batch_tensor, op=torch.distributed.ReduceOp.SUM)
            global_cer = cer_tensor.item() / batch_tensor.item()
            if wer_flag:
                wer_tensor = torch.tensor(total_wer, device=device)
                torch.distributed.all_reduce(wer_tensor, op=torch.distributed.ReduceOp.SUM)
                global_wer = wer_tensor.item() / batch_tensor.item()
                return global_cer, global_wer
            else:
                return global_cer
        else:
            # Single-process evaluation or non-DDP case
            if wer_flag:
                return total_cer / total_batches, total_wer / total_batches
            else:
                return total_cer / total_batches