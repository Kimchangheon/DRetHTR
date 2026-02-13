import os
import multiprocessing
import re

# Define the set of allowed characters
set_all_characters = ''' !"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'''

input_file = 'cc100_random_subset_10M_2_modified.txt'
output_file = 'cc100_random_subset_10M_2_modified_2.txt'

def process_chunk(input_file, output_file, start, end, pattern):
    with open(input_file, 'rb') as infile:
        infile.seek(start)
        if start != 0:
            infile.readline()  # Skip incomplete line
        with open(output_file, 'w', encoding='utf-8') as outfile:
            while True:
                pos = infile.tell()
                if pos >= end:
                    break
                line = infile.readline()
                if not line:
                    break
                line = line.decode('utf-8', errors='ignore').strip()
                # Remove disallowed characters
                new_line = re.sub(pattern, '', line)
                if new_line:
                    outfile.write(new_line + '\n')

def main():
    N = multiprocessing.cpu_count()
    file_size = os.path.getsize(input_file)
    chunk_size = file_size // N
    processes = []

    # Create a regex pattern for character filtering
    allowed_chars_re = re.escape(set_all_characters)
    pattern = f'[^{allowed_chars_re}]'

    for i in range(N):
        start = i * chunk_size
        end = file_size if i == N - 1 else (i + 1) * chunk_size
        p = multiprocessing.Process(
            target=process_chunk,
            args=(input_file, f'{output_file}_{i}', start, end, pattern)
        )
        processes.append(p)

    for p in processes:
        p.start()
    for p in processes:
        p.join()

    # Merge the output files
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for i in range(N):
            part_file = f'{output_file}_{i}'
            with open(part_file, 'r', encoding='utf-8') as infile:
                outfile.write(infile.read())
            os.remove(part_file)

if __name__ == '__main__':
    main()