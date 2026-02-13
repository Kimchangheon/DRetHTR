import os
from fontTools.ttLib import TTFont

def check_font_support(font_path, characters):
    # Load the font
    try:
        font = TTFont(font_path)
        font_chars = set([chr(x) for x in font['cmap'].tables[0].cmap.keys()])
        missing = [char for char in characters if char not in font_chars]
        return missing
    except Exception as e:
        print(f"Error reading font {font_path}: {e}")
        return "load_failed"  # Return a special flag if loading the font fails

def check_fonts_in_directory(directory, characters, output_file):
    missing_fonts = []

    # Loop through all files in the directory
    for file_name in os.listdir(directory):
        # Check if the file is a font (.ttf or .otf)
        if file_name.endswith('.ttf') or file_name.endswith('.otf'):
            font_path = os.path.join(directory, file_name)
            missing = check_font_support(font_path, characters)
            if missing == "load_failed":
                missing_fonts.append(f"{file_name}")
            elif missing:
                missing_fonts.append(f"{file_name}")

    # Write fonts with missing characters or load failures to the output file
    with open(output_file, 'w') as f:
        for font in missing_fonts:
            f.write(f"{font}\n")

    print(f"Fonts with missing characters or load failures written to {output_file}")

# Specify the directory with fonts and the character set to check
font_directory = '/Users/gimchangheon/Documents/GitHub/RetNetHTR/Large_synthetic_data/816_handwritten_Fonts_unzip'  # Change this to the directory path
output_file = 'missing_characters_fonts.txt'
characters = '''!"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'''

# Run the function
check_fonts_in_directory(font_directory, characters, output_file)