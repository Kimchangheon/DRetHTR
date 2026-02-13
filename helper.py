import matplotlib.pyplot as plt
import torch
import os
import re
def get_next_checkpoint_number(run_dir):
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)  # Create directory if it doesn't exist
        return 1

    # Find all checkpoint files in the directory
    checkpoint_files = [f for f in os.listdir(run_dir) if re.match(r'ckpt_\d+\.pt', f)]

    if not checkpoint_files:
        return 1

    # Extract the numeric parts of the checkpoint filenames
    checkpoint_numbers = [int(re.search(r'ckpt_(\d+)\.pt', f).group(1)) for f in checkpoint_files]

    # Return the next available number
    return max(checkpoint_numbers) + 1


def log_memory_usage(stage, device):
    allocated = torch.cuda.memory_allocated(device) / 1024 ** 2  # in MiB
    reserved = torch.cuda.memory_reserved(device) / 1024 ** 2  # in MiB
    peak_allocated = torch.cuda.max_memory_allocated(device) / 1024 ** 2  # Peak allocated memory in MiB
    peak_reserved = torch.cuda.max_memory_reserved(device) / 1024 ** 2  # Peak reserved memory in MiB
    print(f"{stage} - Memory Allocated: {allocated:.2f} MiB, Memory Reserved: {reserved:.2f} MiB, "
          f"Peak Allocated: {peak_allocated:.2f} MiB, Peak Reserved: {peak_reserved:.2f} MiB")
    return peak_allocated

import torch.distributed as dist
def log_memory_usage_ddp(stage, device):
    """Logs GPU memory usage in a distributed setup."""
    device_id = torch.cuda.current_device()  # Get current process's assigned GPU
    allocated = torch.cuda.memory_allocated(device_id) / 1024 ** 2  # in MiB
    reserved = torch.cuda.memory_reserved(device_id) / 1024 ** 2  # in MiB
    peak_allocated = torch.cuda.max_memory_allocated(device_id) / 1024 ** 2  # Peak allocated memory in MiB
    peak_reserved = torch.cuda.max_memory_reserved(device_id) / 1024 ** 2  # Peak reserved memory in MiB

    log_message = (f"GPU {device_id} - {stage}: "
                   f"Allocated: {allocated:.2f} MiB, Reserved: {reserved:.2f} MiB, "
                   f"Peak Allocated: {peak_allocated:.2f} MiB, Peak Reserved: {peak_reserved:.2f} MiB")

    # Print only on rank 0 to avoid redundant logs
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(log_message)

    return peak_allocated

# Function to visualize a single image in the tensor
def visualize_single_image(tensor):
    # Select the specified image and move channel to last dimension for plotting
    img = tensor.permute(1, 2, 0)

    # Normalize the image for better visualization
    img = (img - img.min()) / (img.max() - img.min())

    # Display the image
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def get_last_recorded_epoch(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            if lines:
                last_line = lines[-1]
                # Assume the line starts with "Epoch: <number>"
                last_epoch = int(last_line.split(',')[0].split(':')[1].strip())
                return last_epoch
            else:
                return 0
    except FileNotFoundError:
        # If the file does not exist, start from epoch 0
        return 0
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return 0

# Copied from transformers.models.encoder_decoder.modeling_encoder_decoder.shift_tokens_right
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    if decoder_start_token_id is None:
        raise ValueError("Make sure to set the decoder_start_token_id attribute of the model's configuration.")
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("Make sure to set the pad_token_id attribute of the model's configuration.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids

