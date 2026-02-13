import random
from concurrent.futures import ProcessPoolExecutor

# Global variables
distribution = {}
total_count = 0
percentage_distribution = {}
cumulative_distribution = []

def build_distributions():
    global distribution, total_count, percentage_distribution, cumulative_distribution
    # Step 1: Parse the text_length_counts.txt to get the target distribution
    distribution = {}
    total_count = 0

    with open("text_length_counts.txt", "r") as file:
        next(file)  # Skip header
        for line in file:
            length, count = line.strip().split("\t")
            distribution[int(length)] = int(count)
            total_count += int(count)  # Sum up the total number of lines

    # Step 2: Convert the distribution to percentages (probabilities)
    percentage_distribution = {length: (count / total_count) for length, count in distribution.items()}

    # Step 3: Create a cumulative distribution for selecting a length by probability
    cumulative_distribution = []
    current_cumulative = 0.0
    for length, probability in sorted(percentage_distribution.items()):
        current_cumulative += probability
        cumulative_distribution.append((length, current_cumulative))

# Function to randomly select a length based on cumulative distribution
def select_length_by_probability():
    random_value = random.random()  # Get a random number between 0 and 1
    for length, cumulative in cumulative_distribution:
        if random_value <= cumulative:
            return length
    return max(distribution.keys())  # Return the max length as a fallback

def process_chunk(lines):
    modified_lines = []
    for text in lines:
        text = text.strip()

        if len(text) == 0:
            continue

        selected_length = select_length_by_probability()

        if len(text) > selected_length:
            max_offset = len(text) - selected_length
            offset = random.randint(0, max_offset)
            new_text = text[offset:offset + selected_length]
        elif len(text) < selected_length:
            padding = text * ((selected_length // len(text)) + 1)
            new_text = padding[:selected_length]
        else:
            new_text = text

        modified_lines.append(new_text)
    return modified_lines

def chunked_file_reader(file_path, chunk_size):
    with open(file_path, 'r') as f:
        chunk = []
        for line in f:
            chunk.append(line)
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk

if __name__ == '__main__':
    # Build distributions in the main process
    build_distributions()

    chunk_size = 10000  # Adjust chunk size as needed

    with ProcessPoolExecutor(initializer=build_distributions) as executor, open("cc100_random_subset_10M_2_modified.txt", "w") as out_file:
        for modified_lines in executor.map(process_chunk, chunked_file_reader("cc100_random_subset_10M_2.txt", chunk_size)):
            for line in modified_lines:
                out_file.write(line + "\n")

    print("Processing complete. Modified lines saved to cc100_random_subset_10M_2_modified.txt.")