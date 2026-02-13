import os
import requests
from tqdm import tqdm

def download_file(url, directory):
    # Extract the filename from the URL
    filename = url.split("/")[-1]
    # Create the full path for the download
    file_path = os.path.join(directory, filename)

    # Send a GET request to the URL
    url = "https://www.1001fonts.com"+url
    response = requests.get(url)
    # Check if the request was successful
    if response.status_code == 200:
        # Write the content to a file
        with open(file_path, 'wb') as file:
            file.write(response.content)
        print(f"Downloaded: {file_path}")
    else:
        print(f"Failed to download: {url}")

# URLs of the ZIP files
with open('11954_handwriting_fonts_zip_links.txt', 'r') as file:
    urls = [line.strip() for line in file]

# Directory to save the downloaded files
download_directory = "11954_handwritten_Fonts"

# Create the directory if it doesn't exist
os.makedirs(download_directory, exist_ok=True)

# Download each file
for url in tqdm(urls):
    download_file(url, download_directory)