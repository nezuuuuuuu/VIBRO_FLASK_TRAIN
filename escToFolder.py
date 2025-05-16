import os
import shutil
import csv

def organize_audio_files(csv_file, audio_dir, output_dir):
    """
    Reads a CSV file containing audio file metadata and organizes the
    audio files into subfolders based on their category within a specified
    output directory.

    Args:
        csv_file (str): The path to the input CSV file.
        audio_dir (str): The path to the directory containing the audio files.
        output_dir (str): The path to the directory where the category folders
            should be created.
    """
    # Create the parent output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            filename = row['filename']
            category = row['category']

            # Construct the full path for the source file
            source_path = os.path.join(audio_dir, filename)

            # Create the destination directory if it doesn't exist
            destination_dir = os.path.join(output_dir, category)
            os.makedirs(destination_dir, exist_ok=True)

            # Construct the full path for the destination file
            destination_path = os.path.join(destination_dir, filename)

            try:
                # Move the file
                shutil.move(source_path, destination_path)
                print(f"Moved '{filename}' to '{destination_dir}'")
            except FileNotFoundError:
                print(f"Warning: File '{filename}' not found at '{source_path}'. Skipping.")
            except Exception as e:
                print(f"Error moving '{filename}': {e}")

if __name__ == "__main__":
    # Specify the paths to your CSV file, audio directory, and output directory
    csv_file = 'ESC-50-master/meta/esc50.csv'
    audio_dir = 'ESC-50-master/audio'
    output_dir = 'organized_audio'  # Name of the parent folder

    organize_audio_files(csv_file, audio_dir, output_dir)
import os
import shutil
import csv

def organize_audio_files(csv_file, audio_dir, output_dir):
    """
    Reads a CSV file containing audio file metadata and organizes the
    audio files into subfolders based on their category within a specified
    output directory.

    Args:
        csv_file (str): The path to the input CSV file.
        audio_dir (str): The path to the directory containing the audio files.
        output_dir (str): The path to the directory where the category folders
            should be created.
    """
    # Create the parent output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            filename = row['filename']
            category = row['category']

            # Construct the full path for the source file
            source_path = os.path.join(audio_dir, filename)

            # Create the destination directory if it doesn't exist
            destination_dir = os.path.join(output_dir, category)
            os.makedirs(destination_dir, exist_ok=True)

            # Construct the full path for the destination file
            destination_path = os.path.join(destination_dir, filename)

            try:
                # Move the file
                shutil.move(source_path, destination_path)
                print(f"Moved '{filename}' to '{destination_dir}'")
            except FileNotFoundError:
                print(f"Warning: File '{filename}' not found at '{source_path}'. Skipping.")
            except Exception as e:
                print(f"Error moving '{filename}': {e}")

if __name__ == "__main__":
    # Specify the paths to your CSV file, audio directory, and output directory
    csv_file = 'ESC-50-master/meta/esc50.csv'
    audio_dir = 'ESC-50-master/audio'
    output_dir = 'organized_audio'  # Name of the parent folder

    organize_audio_files(csv_file, audio_dir, output_dir)
