from collections import Counter
import matplotlib.pyplot as plt

# Initialize an empty list to store the lengths of each text
text_lengths = []

# Open and read the file
with open("IAM_A_images_with_transcriptions.txt", "r") as file:
    for line in file:
        # Split each line by the tab character to get the image name and the corresponding text
        try:
            _, text = line.strip().split("\t")
            # Calculate the number of words in the text (split by spaces)
            length = len(text)
            # Add the length to the list
            text_lengths.append(length)
        except ValueError:
            # Handle any lines that might not be formatted correctly
            print(f"Skipping malformed line: {line}")

# Calculate the distribution of text lengths
length_distribution = Counter(text_lengths)

# Sort the lengths and corresponding counts
lengths = sorted(length_distribution.keys())
counts = [length_distribution[length] for length in lengths]

# Plot the distribution
plt.figure(figsize=(12, 8))
plt.bar(lengths, counts, color='skyblue')

# Add labels and title
plt.xlabel('Text Length (Number of Characters)')
plt.ylabel('Frequency')
plt.title('Distribution of Text Lengths IAM lines (Max Length = 94)')

# Save the plot as a PNG file
plt.savefig('text_length_distribution.png', format='png')

# Save the counts to a text file
with open("text_length_counts.txt", "w") as count_file:
    count_file.write("Text Length\tFrequency\n")
    for length, count in sorted(length_distribution.items()):
        count_file.write(f"{length}\t{count}\n")

# Show the plot (optional)
plt.show()