import os
from IPython import display
import tensorflow as tf
import tensorflow_io as tfio
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import os
os.environ["TFHUB_CACHE_DIR"] = "./my_tfhub_cache"

def create_tflite_model_from_csv(esc50_csv, base_data_path, modelName, yamnet_model_handle="basemodel"):
    """
    Creates a TensorFlow Lite model from a CSV file containing audio file paths and labels.

    Args:
        esc50_csv (str): Path to the CSV file.
        base_data_path (str): Base path where audio files are located.
        modelName (str): Name to use when saving the TFLite model.
        yamnet_model_handle (str, optional): Handle for the YAMNet model.
            Defaults to "basemodel".

    Returns:
        tuple: (tflite_file_path, custom_labels) if successful, (None, None) otherwise.
            tflite_file_path: Path to the saved TFLite model.
            custom_labels: A list of the custom class names.
    """
    try:
        # Load YAMNet model
        yamnet_model = hub.load(yamnet_model_handle)

        # Utility functions for loading audio files
        @tf.function
        def load_wav_16k_mono(filename):
            file_contents = tf.io.read_file(filename)
            wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
            wav = tf.squeeze(wav, axis=-1)
            sample_rate = tf.cast(sample_rate, dtype=tf.int64)
            wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
            return wav

        def load_wav_for_map(filename, label, fold):
            return load_wav_16k_mono(filename), label, fold

        def extract_embedding(wav_data, label, fold):
            scores, embeddings, spectrogram = yamnet_model(wav_data)
            num_embeddings = tf.shape(embeddings)[0]
            return (embeddings,
                    tf.repeat(label, num_embeddings),
                    tf.repeat(fold, num_embeddings))

        # Load class names from YAMNet
        class_map_path = yamnet_model.class_map_path().numpy().decode('utf-8')
        class_names = list(pd.read_csv(class_map_path)['display_name'])

        # Load and preprocess the CSV data using pandas
        pd_data = pd.read_csv(esc50_csv)

        # --- Adapt this part to filter your data ---
        # Example filtering: Keep only categories that actually exist.
        all_classes = pd_data['category'].unique()
        map_class_to_id = {category: i for i, category in enumerate(all_classes)}

        filtered_pd = pd_data.assign(
            target=pd_data['category'].map(map_class_to_id)
        )

        full_path = filtered_pd['filename'].apply(lambda row: os.path.join(base_data_path, row))
        filtered_pd = filtered_pd.assign(filename=full_path)

        # Create TensorFlow Dataset
        filenames = filtered_pd['filename']
        targets = filtered_pd['target']
        folds = filtered_pd['fold']
        main_ds = tf.data.Dataset.from_tensor_slices((filenames, targets, folds))

        # Preprocess the dataset: load audio, extract embeddings
        main_ds = main_ds.map(load_wav_for_map)
        main_ds = main_ds.map(extract_embedding).unbatch()

        # Split the dataset into training, validation, and test sets
        cached_ds = main_ds.cache()
        train_ds = cached_ds.filter(lambda embedding, label, fold: fold < 2)
        val_ds = cached_ds.filter(lambda embedding, label, fold: fold == 3)
        test_ds = cached_ds.filter(lambda embedding, label, fold: fold == 4)

        # Remove the folds column
        remove_fold_column = lambda embedding, label, fold: (embedding, label)
        train_ds = train_ds.map(remove_fold_column)
        val_ds = val_ds.map(remove_fold_column)
        test_ds = test_ds.map(remove_fold_column)

        # Configure data for training
        train_ds = train_ds.cache().shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.cache().batch(32).prefetch(tf.data.AUTOTUNE)
        test_ds = test_ds.cache().batch(32).prefetch(tf.data.AUTOTUNE)

        # Define the model
        my_classes = all_classes  # Use the classes derived from the CSV
        custom_labels = list(my_classes) # <-- Get the custom labels
        my_model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(1024), dtype=tf.float32, name='input_embedding'),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(254, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(len(my_classes), activation='softmax', )
        ], name='my_model')


        my_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                        metrics=['accuracy'])

        callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=1
        )
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=1, factor=0.1, min_lr=1e-7)

        with tf.device('/GPU:0'):  # Or specify your GPU
            history = my_model.fit(train_ds,
                                    epochs=20,
                                    validation_data=val_ds,
                                    callbacks=[callback, lr_scheduler])

        # ---  Adapt the following part for the new model ---
        class ReduceMeanLayer(tf.keras.layers.Layer):
            def __init__(self, axis=0, **kwargs):
                super(ReduceMeanLayer, self).__init__(**kwargs)
                self.axis = axis

            def call(self, input):
                return tf.math.reduce_mean(input, axis=self.axis)

        # Save the trained model
        saved_model_path = './' + modelName  # Changed to a constant
        input_segment = tf.keras.layers.Input(shape=(), dtype=tf.float32, name='audio')
        embedding_extraction_layer = hub.KerasLayer(yamnet_model_handle,
                                                    trainable=False, name='yamnet')
        score, embeddings_output, _ = embedding_extraction_layer(input_segment)
        serving_outputs = my_model(embeddings_output)
        serving_outputs = ReduceMeanLayer(axis=0, name='classifier')(serving_outputs)
        averaged_scores = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=0), name='avg_yamnet_scores')(score)
        serving_model = tf.keras.Model(input_segment,
                                        outputs={
                                            'yamnet_scores': averaged_scores,
                                            'custom_classification': serving_outputs
                                        }
                                        )
        serving_model.save(saved_model_path, include_optimizer=False)

        # Convert the model to TFLite
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
        tflite_model = converter.convert()

        # Save the TFLite model to a file.  Use a constant filename.
        tflite_file_path = f'{modelName}.tflite'
        with open(tflite_file_path, 'wb') as f:
            f.write(tflite_model)

        print(f"✅ TFLite model saved as {tflite_file_path}")
        return tflite_file_path, custom_labels  # Return the path to the TFLite model and labels

    except Exception as e:
        print(f"Error creating TFLite model: {e}")
        return None, None  # Return None to indicate failure
    
if __name__ == '__main__':
    # Example usage:
    esc50_csv_path = 'local_folders\metadata.csv'
    base_data_path = ''

    tflite_model_path, custom_labels = create_tflite_model_from_csv(esc50_csv_path, base_data_path,"group_model")
    if tflite_model_path:
        print(f"TFLite model saved successfully at: {tflite_model_path}")
        print(f"Custom labels: {custom_labels}")
    else:
        print("Failed to create TFLite model.")
