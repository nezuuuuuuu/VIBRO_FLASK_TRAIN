from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from pymongo import MongoClient
from bson import ObjectId
import os
import base64
import wave
import csv
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
import soundfile as sf
import shutil
import os
import csv
import random
import os
from IPython import display
import tensorflow as tf
import tensorflow_io as tfio
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import os
from train import create_tflite_model_from_csv
from flask import send_file
import datetime # Import datetime
app = Flask(__name__)
CORS(app)

PORT = 5000
MONGO_URI = 'mongodb+srv://johnmarkeconar7:HEWlz7E3htnP6dKt@cluster0.wxvsc4q.mongodb.net/vibro_db?retryWrites=true&w=majority&appName=Cluster0'

augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=1.0),
    TimeStretch(min_rate=0.9, max_rate=1.1, p=1.0),
    PitchShift(min_semitones=-2, max_semitones=2, p=1.0),
    Shift(min_shift=-0.1, max_shift=0.1, p=1.0)
])

client = MongoClient(MONGO_URI)
db = client["vibro_db"]
sounds_collection = db["customsounds"]
folders_collection = db["customfolders"]
group_collection = db["groups"]
model_collection = db["models"] # Add model collection

def create_folder_metadata_csv(base_folder_path):
    """
    Generates a CSV metadata file for a given base folder.

    The CSV includes 'filename', 'fold', 'target', and 'category' columns.
    'category' is the folder name, 'filename' includes the full path,
    'target' is the folder's index, and 'fold' is a random value from 0-4.
    The rows are shuffled after fold assignment.

    Args:
        base_folder_path (str): The path to the base folder containing subfolders.
    """
    csv_rows = []
    
    # Get subfolders (categories)
    subfolders = [f.name for f in os.scandir(base_folder_path) if f.is_dir()]
    subfolders.sort() # Ensure consistent ordering for target index

    for target_index, category in enumerate(subfolders):
        category_path = os.path.join(base_folder_path, category)
        for filename in os.listdir(category_path):
            full_filepath = os.path.join(category_path, filename)
            if os.path.isfile(full_filepath): # Ensure it's a file, not a sub-directory
                csv_rows.append({
                    'filename': full_filepath,
                    'fold': random.randint(0, 4),  # Assign a random fold
                    'target': target_index,
                    'category': category
                })

    # Shuffle the rows after assigning folds
    random.shuffle(csv_rows)

    # Write CSV file
    csv_path = os.path.join(base_folder_path, "metadata.csv")
    try:
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['filename', 'fold', 'target', 'category'])
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"Saved metadata CSV at: {csv_path}")
    except Exception as e:
        print(f"Error writing CSV: {e}")

def copy_folder_contents(source_folder, destination_folder):
    """
    Copies the contents of a folder (source_folder) to another folder (destination_folder).
    If the destination folder does not exist, it will be created.  Files and
    subdirectories within the source folder are copied recursively.

    Args:
        source_folder (str): The path to the folder to copy contents from.
        destination_folder (str): The path to the folder to copy contents to.
    """
    try:
        # Ensure source folder exists
        if not os.path.exists(source_folder):
            print(f"Error: Source folder '{source_folder}' does not exist.")
            return

        # Create destination folder if it doesn't exist
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)
            print(f"Created destination folder: {destination_folder}")

        # Use shutil.copytree to copy contents recursively
        for item in os.listdir(source_folder):
            source_item_path = os.path.join(source_folder, item)
            dest_item_path = os.path.join(destination_folder, item)
            if os.path.isdir(source_item_path):
                shutil.copytree(source_item_path, dest_item_path, dirs_exist_ok=True)
                print(f"Copied directory: {source_item_path} to {dest_item_path}")
            else:
                shutil.copy2(source_item_path, dest_item_path)  # copy2 preserves metadata
                print(f"Copied file: {source_item_path} to {dest_item_path}")

        print(f"Successfully copied contents from '{source_folder}' to '{destination_folder}'")

    except Exception as e:
        print(f"An error occurred during the copy process: {e}")

if __name__ == "__main__":
    # Example usage:
    source_folder = "source_folder"  # Replace with your source folder path
    destination_folder = "destination_folder"  # Replace with your destination folder path

    # Create dummy source folder and files for testing
    if not os.path.exists(source_folder):
        os.makedirs(source_folder)
        with open(os.path.join(source_folder, "file1.txt"), "w") as f:
            f.write("This is file 1.")
        with open(os.path.join(source_folder, "file2.txt"), "w") as f:
            f.write("This is file 2.")
        os.makedirs(os.path.join(source_folder, "subdir"))
        with open(os.path.join(source_folder, "subdir", "file3.txt"), "w") as f:
            f.write("This is file 3 in subdir.")

    copy_folder_contents(source_folder, destination_folder)
    

def serialize_doc(doc):
    if isinstance(doc, list):
        return [serialize_doc(item) for item in doc]
    elif isinstance(doc, dict):
        return {key: serialize_doc(value) for key, value in doc.items()}
    elif isinstance(doc, ObjectId):
        return str(doc)
    else:
        return doc

@app.route('/folders', methods=['GET'])
def get_folders():
    group_id = request.args.get('groupId')
    if not group_id:
        return jsonify({'error': 'Missing groupId parameter'}), 400

    try:
        group_object_id = ObjectId(group_id)
    except Exception:
        return jsonify({'error': 'Invalid groupId format'}), 400

    folders = list(folders_collection.find({'groupId': group_object_id}, {'_id': 1, 'folderName': 1}))
    serialized_folders = [serialize_doc(folder) for folder in folders]
    group_data = group_collection.find_one({'_id': group_object_id}, {'groupName': 1, '_id': 0})
    group_name = group_data.get('groupName') if group_data else None
    base_path = './local_folders'
    # os.makedirs(base_path, exist_ok=True)
    if os.path.exists(base_path):
        print(f"Deleting existing folder: '{base_path}'")
        shutil.rmtree(base_path)
    
    os.makedirs(base_path)

    csv_rows = []
    category_to_index = {}
    current_index = 0

    for folder in serialized_folders:
        folder_name = folder.get('folderName')
        folder_id = folder.get('_id')

        if not (folder_name and folder_id):
            continue

        # Assign numeric index to category
        if folder_name not in category_to_index:
            category_to_index[folder_name] = current_index
            current_index += 1
        target = category_to_index[folder_name]

        # Make folder path safe
        safe_folder_name = "".join(c for c in folder_name if c.isalnum() or c in (' ', '_', '-')).rstrip()
        folder_path = os.path.join(base_path, safe_folder_name)
        os.makedirs(folder_path, exist_ok=True)

        sounds = list(sounds_collection.find({'folderId': ObjectId(folder_id)}, {'_id': 0, 'sound': 1, 'filename': 1}))
        
        for sound_doc in sounds:
            filename = sound_doc.get('filename')
            sound_data = sound_doc.get('sound')

            if not (filename and sound_data):
                continue

            try:
                clean_base64 = "".join(sound_data.split())
                pcm_bytes = base64.b64decode(clean_base64)
            except Exception as e:
                print(f"Error decoding base64 for {filename}: {e}")
                continue

            if not filename.lower().endswith('.wav'):
                filename += '.wav'
            file_path = os.path.join(folder_path, filename)

            # Write original WAV file
            try:
                with wave.open(file_path, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(16000)
                    wf.writeframes(pcm_bytes)
                print(f"Saved WAV file: {file_path}")
                csv_rows.append({
                    'filename': filename,
                    'fold': 1,
                    'target': target,
                    'category': folder_name
                })
            except Exception as e:
                print(f"Error writing WAV file {file_path}: {e}")
                continue

            # Augmentation
            try:
                samples, sample_rate = sf.read(file_path)
                for i in range(1, 6):
                    augmented_samples = augment(samples=samples, sample_rate=sample_rate)
                    augmented_filename = f"{filename[:-4]}_aug{i}.wav"
                    augmented_path = os.path.join(folder_path, augmented_filename)
                    sf.write(augmented_path, augmented_samples, sample_rate)
                    print(f"Saved augmented file: {augmented_path}")
                    
            except Exception as e:
                    print(f"Error augmenting audio {file_path}: {e}")

    copy_folder_contents("organized_audio", base_path)
    create_folder_metadata_csv(base_path)
    esc50_csv_path = 'local_folders\metadata.csv'
    base_data_path = ''

    tflite_model_path, custom_labels = create_tflite_model_from_csv(esc50_csv_path, base_data_path,"group_model")
    if tflite_model_path and os.path.exists(tflite_model_path):
        # Create a temporary file for labels
        labels_file_name = f'{group_name}_labels.txt'
        labels_file_path = os.path.join(base_path, labels_file_name)
        with open(labels_file_path, 'w') as f:
            for label in custom_labels:
                f.write(f"{label}\n")
        
        #save model document
        model_data = {
        'groupId': group_object_id,
        'modelName': group_name,
        'modelLabels': custom_labels,
        'labelCount': len(custom_labels),
        'filePath': tflite_model_path,
        'timestamp': datetime.datetime.utcnow()
        }

        model_collection.replace_one(
            {'groupId': group_object_id},  # match by groupId
            model_data,
            upsert=True
        )
        return send_file(
            tflite_model_path,
            as_attachment=True,
            download_name=f'{group_name}.tflite',  # the file name clients will receive
            mimetype='application/octet-stream'
        )
    else:
        return jsonify({"message": "Failed to create TFLite model"}), 500

    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT)
