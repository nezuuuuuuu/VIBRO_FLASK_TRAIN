from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from pymongo import MongoClient
from bson import ObjectId
import urllib.parse
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
from train_multilabel import create_tflite_model_from_csv
from flask import send_file
import datetime # Import datetime
import boto3
from dotenv import load_dotenv
load_dotenv()
import os
import csv
import numpy as np
from collections import defaultdict
load_dotenv()




aws_access_key_id = os.getenv("aws_access_key_id")
aws_secret_access_key = os.getenv("zXenRt6g0RnwirnE2pOrd4YUfVr461AT0TByXlvm")
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









# Set the root directory
root_dir = 'local_folders'
csv_file = 'local_folders\metadata.csv'
num_folds = 5
all_class_names = None


def create_class_name_mapping(root_dir, classes_to_include=None):
    folder_names = sorted([
        folder for folder in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, folder))
    ])
    if classes_to_include:
        folder_names = [f for f in folder_names if f in classes_to_include]
    folder_to_index = {class_name: i for i, class_name in enumerate(folder_names)}
    return folder_to_index, folder_names



def create_folder_metadata_csv(base_folder_path):



    # Create mapping
    folder_to_index, all_class_names = create_class_name_mapping(root_dir)
    num_classes = len(all_class_names)

    # Dictionary: filename → (filepath, label_vector)
    file_label_map = {}

    # Collect all valid files
    for subdir, dirs, files in os.walk(root_dir):
        label = os.path.basename(subdir)
        if label not in folder_to_index:
            continue  # Skip if label not included

        class_index = folder_to_index[label]
        
        for file in files:
            if not file.endswith('.wav'):
                continue
            filepath = os.path.join(subdir, file).replace("\\", "/")

            if file not in file_label_map:
                label_vector = [0] * num_classes
                file_label_map[file] = (filepath, label_vector)
            
            _, label_vector = file_label_map[file]
            label_vector[class_index] = 1  # Add label

    # Now, filter out `Speech`-only and `Animal`-only files if they exceed the limit
    filtered_items = []
    for filename, (filepath, label_vector) in file_label_map.items():
            filtered_items.append((filename, filepath, label_vector))



    # Shuffle the filtered entries
    np.random.shuffle(filtered_items)

    # Write CSV
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header = ['filepath', 'fold'] + all_class_names
        writer.writerow(header)

        for idx, (filename, filepath, label_vector) in enumerate(filtered_items):
            fold_index = idx % num_folds
            row = [filepath, fold_index] + label_vector
            writer.writerow(row)



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
            # print(f"Created destination folder: {destination_folder}")

        # Use shutil.copytree to copy contents recursively
        for item in os.listdir(source_folder):
            source_item_path = os.path.join(source_folder, item)
            dest_item_path = os.path.join(destination_folder, item)
            if os.path.isdir(source_item_path):
                shutil.copytree(source_item_path, dest_item_path, dirs_exist_ok=True)
                # print(f"Copied directory: {source_item_path} to {dest_item_path}")
            else:
                shutil.copy2(source_item_path, dest_item_path)  # copy2 preserves metadata
                # print(f"Copied file: {source_item_path} to {dest_item_path}")

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

def update_group_model_url(group_id, group_collection,status):
    # 1. Check if group_id is missing (Your existing logic)
    if not group_id:
        return jsonify({'error': 'Missing groupId parameter'}), 400
    
    # 2. If group_id exists, proceed to update the document
    else:
        try:
            # Convert the string group_id to a MongoDB ObjectId
            group_object_id = ObjectId(group_id) 
            
            # Perform the update operation
            result = group_collection.update_one(
                # Query/Filter: Find the document by its _id
                {'_id': group_object_id},
                {'$set': {'groupModelUrl': status}}
            )
            
            # Check the outcome of the update
            if result.matched_count == 0:
                # No document found with that ID
                return jsonify({'error': f'Group with ID {group_id} not found'}), 404
            
            # Success
            return jsonify({
                'message': 'Group model URL updated successfully',
                'id': group_id,
                'status': status
            }), 200

        except Exception as e:
            # Handle cases where the ID is invalid (e.g., not a valid ObjectId string)
            return jsonify({'error': f'Invalid group ID format: {e}'}), 400
@app.route('/folders', methods=['GET'])
def get_folders():
    group_id = request.args.get('groupId')
    if not group_id:
        return jsonify({'error': 'Missing groupId parameter'}), 400
  
    try:
        
        group_object_id = ObjectId(group_id)
        update_group_model_url(group_object_id, group_collection,"PENDING")
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
                    # print(f"Saved augmented file: {augmented_path}")
                    
            except Exception as e:
                    print(f"Error augmenting audio {file_path}: {e}")

    copy_folder_contents("organized_audio", base_path)
    csv_path = 'local_folders\metadata.csv'
    base_data_path = ''
    folder_to_index, all_class_names = create_class_name_mapping(root_dir)
    create_folder_metadata_csv(base_path)
    print("size of the class", len(all_class_names))
    tflite_model_path= create_tflite_model_from_csv(csv_path, base_data_path,"group_model",(len(all_class_names)))
   
    if tflite_model_path and os.path.exists(tflite_model_path):
        # Create a temporary file for labels
        labels_file_name = f'{group_name}_labels.txt'
        labels_file_path = os.path.join(base_path, labels_file_name)
        with open(labels_file_path, 'w') as f:
            for label in all_class_names:
                f.write(f"{label}\n")
        
        #save model document
        model_data = {
        'groupId': group_object_id,
        'modelName': group_name,
        'modelLabels': all_class_names,
        'labelCount': len(all_class_names),
        'filePath': tflite_model_path,
        'timestamp': datetime.datetime.utcnow()
        }

        model_collection.replace_one(
            {'groupId': group_object_id},  # match by groupId
            model_data,
            upsert=True
        )
        s3 = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name='ap-southeast-2'
        )
    
        BUCKET_NAME = "vibro-models"
        REGION_CODE = "ap-southeast-2"
        OBJECT_KEY = f"{group_name}.tflite"

        safe_key = urllib.parse.quote(OBJECT_KEY, safe="~()*!.'")


        download_url = f"https://{BUCKET_NAME}.s3.{REGION_CODE}.amazonaws.com/{safe_key}"

        s3.upload_file(tflite_model_path, 
                        "vibro-models", 
                        f"{group_name}.tflite",
                        ExtraArgs={'ACL': 'public-read'}
            )
        update_group_model_url(group_object_id, group_collection,download_url)
        # return send_file(
        #     tflite_model_path,
        #     as_attachment=True,
        #     download_name=f'{group_name}.tflite',  # the file name clients will receive
        #     mimetype='application/octet-stream'
        # )
    else:
        return jsonify({"message": "Failed to create TFLite model"}), 500

    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT)
