import multiprocessing as mp


# Function to process each chunk and find duplicates
def process_chunk(chunk):
    unique_lines = set()
    duplicates = set()

    for line in chunk:
        stripped_line = line.strip()
        if stripped_line in unique_lines:
            duplicates.add(stripped_line)
        else:
            unique_lines.add(stripped_line)

    return duplicates


# Function to divide the file into chunks
def chunkify(file_path, num_chunks):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    chunk_size = len(lines) // num_chunks
    for i in range(0, len(lines), chunk_size):
        yield lines[i:i + chunk_size]


def main():
    file_path = 'cc100_random_subset_10M_2_modified_2.txt'
    num_workers = mp.cpu_count()  # Number of parallel workers

    # Create a pool of workers
    with mp.Pool(num_workers) as pool:
        # Split the file into chunks and process them in parallel
        chunked_results = pool.map(process_chunk, chunkify(file_path, num_workers))

    # Combine results from all chunks
    all_duplicates = set()
    for result in chunked_results:
        all_duplicates.update(result)

    # Output the results
    if all_duplicates:
        print(f"Duplicate lines found: {len(all_duplicates)}") #5405 --> 1230
        print(len(all_duplicates))
        for dup in all_duplicates:
            print(dup)
    else:
        print("No duplicate lines found.")


if __name__ == '__main__':
    main()