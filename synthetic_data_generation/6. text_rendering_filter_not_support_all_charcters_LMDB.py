import os
import random
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, Manager
from fontTools.ttLib import TTFont
import io
import lmdb
import pickle

# Directory containing font files
font_dir = '11954_handwritten_Fonts_all'  # Replace with your font directory path

output_dir = '/home/vault/b193dc/b193dc11/'
os.makedirs(output_dir, exist_ok=True)

# Get list of font files with extensions .ttf, .otf, etc.
font_files = []
for root, dirs, files in os.walk(font_dir):
    for file in files:
        if file.lower().endswith(('.otf', '.ttf')):
            font_files.append(os.path.join(root, file))

# Read the text lines from your text file
with open('cc100_random_subset_10M_modified_2.txt', 'r') as file:
    texts = file.readlines()

def check_font_support(font_path, characters):
    # Load the font
    try:
        font = TTFont(font_path)
        font_cmaps = []
        for table in font['cmap'].tables:
            font_cmaps.extend(table.cmap.keys())
        font_chars = set(chr(codepoint) for codepoint in font_cmaps)
        missing = [char for char in characters if char not in font_chars]
        return not missing  # True if no missing characters
    except Exception as e:
        print(f"Error reading font {font_path}: {e}")
        return False  # Return False if loading the font fails

def generate_image(args):
    i, text, font_files_global = args
    text = text.strip()
    # font_files = font_files_global.copy()
    font_files = list(font_files_global)

    font_loaded = False
    attempts = 0
    max_attempts = len(font_files)

    # Attempt to load a font that supports all characters
    while not font_loaded and attempts < max_attempts:
        font_path = random.choice(font_files)
        font_files.remove(font_path)
        if check_font_support(font_path, text):
            try:
                font_size = 100
                font = ImageFont.truetype(font_path, font_size)
                font_loaded = True
            except Exception as e:
                print(f"Failed to load font: {font_path}. Error: {e}")
                attempts += 1
        else:
            attempts += 1

    if not font_loaded:
        print(f"Could not find a font that supports all characters in text '{text}'. Skipping index {i}.")
        return None  # Return None to indicate failure

    if not text:
        print(f"Text is empty after stripping. Skipping index {i}.")
        return None

    # Create a dummy image to get a drawing context
    dummy_img = Image.new('RGB', (1, 1))
    draw = ImageDraw.Draw(dummy_img)
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
    except Exception as e:
        print(f"Error getting textbbox for text '{text}': {e}")
        return None

    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    if text_width <= 0 or text_height <= 0:
        print(f"Text '{text}' results in non-positive dimensions ({text_width}x{text_height}). Skipping.")
        return None

    # Create a new image with the size of the text
    img = Image.new('RGB', (text_width, text_height), color='white')
    draw = ImageDraw.Draw(img)
    draw.text((-bbox[0], -bbox[1]), text, font=font, fill='black')

    # Convert image to bytes in PNG format
    try:
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='PNG')
        img_bytes = img_buffer.getvalue()
    except Exception as e:
        print(f"Error saving image to buffer for index {i}: {e}")
        return None

    return (i, img_bytes, text)

def worker_initializer(font_files_global_):
    # Make font_files_global accessible in worker processes
    global font_files_global
    font_files_global = font_files_global_

if __name__ == '__main__':
    from multiprocessing import Manager

    manager = Manager()
    font_files_global = manager.list(font_files)

    # Prepare the inputs for multiprocessing
    inputs = [(i, text, font_files_global) for i, text in enumerate(texts)]

    lmdb_output_path = 'synthesized_images_10M.lmdb'
    env = lmdb.open(lmdb_output_path, map_size=1e12)

    # Function to write data to LMDB incrementally
    def process_and_write(args):
        result = generate_image(args)
        if result is not None:
            i, img_bytes, text = result
            with env.begin(write=True) as txn:
                key = f'{i:07}'.encode('ascii')
                data = {'image': img_bytes, 'text': text}
                txn.put(key, pickle.dumps(data))

    with Pool(cpu_count()) as pool:
        list(tqdm(pool.imap_unordered(process_and_write, inputs), total=len(inputs)))

    env.close()