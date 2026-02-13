from datasets import load_dataset
import random
import argparse
from multiprocessing import Pool
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Train Retnet2Retnet with specified parameters.')
    parser.add_argument('--split', type=str, default="train", help='split type')
    return parser.parse_args()

args = parse_args()

def init_worker(lang, split, cache_dir):
    global dataset
    dataset = load_dataset('cc100', lang, split=split, cache_dir=cache_dir)
    # Disable caching to prevent disk space issues
    dataset.set_format(type=None)

def worker_process(args):
    indices, temp_output_file, temp_index_file = args
    with open(temp_output_file, 'w', encoding='utf-8') as outfile, \
         open(temp_index_file, 'w', encoding='utf-8') as indexfile:
        for idx in indices:
            item = dataset[idx]
            outfile.write(item['text'] + '\n')
            indexfile.write(str(idx) + '\n')

def extract_random_lines_from_hf_dataset(num_lines=10000000, lang='en',
                                         output_file='cc100_random_subset_10M_2.txt',
                                         index_file='cc100_random_indices_10M_2.txt',
                                         num_processes=2, cache_dir="/path/to/cache"):
    # Load the dataset once to get the size
    dataset = load_dataset('cc100', lang, split=args.split, cache_dir=cache_dir)
    dataset_size = len(dataset)

    exclue_prev = True
    if exclue_prev :
        # Load used indices from the index file
        used_indices = set()
        with open('cc100_random_indices_10M.txt', 'r') as f:
            for line in f:
                used_indices.add(int(line.strip()))

        # Generate new random indices, excluding used indices
        remaining_indices = set(range(dataset_size)) - used_indices
        if len(remaining_indices) < num_lines:
            raise ValueError("Not enough unused indices to sample the requested number of lines.")

        random_indices = random.sample(remaining_indices, num_lines)

    else :
        # Generate random indices
        random_indices = random.sample(range(dataset_size), num_lines)
        random_indices

    # Split indices among processes
    chunk_size = num_lines // num_processes
    index_chunks = [random_indices[i*chunk_size:(i+1)*chunk_size] for i in range(num_processes)]
    # Add remaining indices to the last chunk
    index_chunks[-1].extend(random_indices[num_processes*chunk_size:])

    # Prepare arguments for worker processes
    temp_files = []
    args_list = []
    for i, indices in enumerate(index_chunks):
        temp_output_file = f'{output_file}_part_{i}'
        temp_index_file = f'{index_file}_part_{i}'
        temp_files.append((temp_output_file, temp_index_file))
        args_list.append((indices, temp_output_file, temp_index_file))

    # Initialize worker processes
    with Pool(processes=num_processes, initializer=init_worker, initargs=(lang, args.split, cache_dir)) as pool:
        pool.map(worker_process, args_list)

    # Merge temporary files
    with open(output_file, 'w', encoding='utf-8') as outfile, \
         open(index_file, 'w', encoding='utf-8') as indexfile:
        for temp_output_file, temp_index_file in temp_files:
            with open(temp_output_file, 'r', encoding='utf-8') as infile:
                outfile.write(infile.read())
            os.remove(temp_output_file)

            with open(temp_index_file, 'r', encoding='utf-8') as infile:
                indexfile.write(infile.read())
            os.remove(temp_index_file)

    print(f'{num_lines} random lines have been extracted to {output_file}')
    print(f'Random indices have been saved to {index_file}')

# Example usage
extract_random_lines_from_hf_dataset(num_processes=os.cpu_count(), cache_dir="/disks/data2/no12neni/.cache/huggingface/datasets")