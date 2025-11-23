import os
from IPython import display
import tensorflow as tf
import tensorflow_io as tfio
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import os
os.environ["TFHUB_CACHE_DIR"] = "./my_tfhub_cache"

def create_tflite_model_from_csv(esc50_csv, base_data_path, modelName, folder_count, yamnet_model_handle="basemodel"):


    try:
        # Load YAMNet model
        yamnet_model = hub.load(yamnet_model_handle)

        # Utility functions for loading audio files
        @tf.function
        def load_wav_16k_mono(filename):
            """ Load a WAV file, convert it to a float tensor, resample to 16 kHz single-channel audio. """
            file_contents = tf.io.read_file(filename)
            wav, sample_rate = tf.audio.decode_wav(
                file_contents,
                desired_channels=1)
            wav = tf.squeeze(wav, axis=-1)
            sample_rate = tf.cast(sample_rate, dtype=tf.int64)
            wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
            return wav

       
        def extract_embedding(wav_data, label, fold):
            scores, embeddings, spectrogram = yamnet_model(wav_data)

            num_frames = tf.shape(embeddings)[0]
            # Ensure label has shape [num_classes] -> expand to [1, num_classes]
            label = tf.expand_dims(label, axis=0)  # [1, num_classes]
            labels = tf.tile(label, [num_frames, 1])  # [num_frames, num_classes]

            folds = tf.repeat(fold, repeats=num_frames)
            return embeddings, labels, folds
        print("YAMNet model loaded successfully.")

        # Load and preprocess the CSV data using pandas
        pd_data = pd.read_csv(esc50_csv)
     
        labels = []
        for index, row in pd_data.iterrows():
            labels.append(list(row.iloc[-folder_count:]))

        audio_path = pd_data['filepath'].to_list
        audio_path = pd_data['filepath'].to_list()

        # Add './' to the start of each path
        audio_path = ['./' + path for path in audio_path]

        filenames = pd_data['filepath'].tolist()
        targets = labels
        folds = pd_data['fold'].tolist()
        main_ds = tf.data.Dataset.from_tensor_slices((filenames, targets, folds))

     
        def load_wav_for_map(filename, labels, fold):
            return load_wav_16k_mono(filename), labels, fold

        main_ds = main_ds.map(load_wav_for_map)
      

        # extract embedding
        main_ds = main_ds.map(extract_embedding).unbatch()
        print(main_ds.element_spec)

        # Cache dataset after embedding extraction
        cached_ds = main_ds.cache()
        cached_ds = main_ds.cache("cache.tf-data")  # Will create a cache file on disk
        print("Dataset cached successfully.")
        # Filter splits using fold
        train_ds = cached_ds.filter(lambda emb, label, fold: fold < 3)
        val_ds   = cached_ds.filter(lambda emb, label, fold: fold == 3)
        test_ds  = cached_ds.filter(lambda emb, label, fold: fold == 4)

        # Remove fold column (keep embedding and label only)
        def remove_fold_column(embedding, label, fold):
            return embedding, label

        # Apply to all splits
        train_ds = train_ds.map(remove_fold_column, num_parallel_calls=tf.data.AUTOTUNE)
        val_ds   = val_ds.map(remove_fold_column, num_parallel_calls=tf.data.AUTOTUNE)
        test_ds  = test_ds.map(remove_fold_column, num_parallel_calls=tf.data.AUTOTUNE)


        # Final batching, shuffling, and prefetching
        train_ds = train_ds.shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
        val_ds   = val_ds.batch(32).prefetch(tf.data.AUTOTUNE)
        test_ds  = test_ds.batch(32).prefetch(tf.data.AUTOTUNE)

        from tensorflow.keras import regularizers


        my_model = tf.keras.Sequential([
            tf.keras.layers.Dense(1024, activation='relu', input_shape=(1024,), kernel_regularizer=regularizers.l2(0.001)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.0005)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(folder_count, activation='sigmoid')
        ], name='regularized_model')
        
        print(my_model.summary())
        my_model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                 optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                 metrics=['accuracy'])
        from tensorflow.keras.callbacks import ReduceLROnPlateau

        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', patience=1, factor=0.1, min_lr=1e-7)

        callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,         
            restore_best_weights=True,
            verbose=1          
)
        with tf.device('/GPU:0'):  # Change to '/GPU:1', '/GPU:2', etc., if you have multiple GPUs=
            history = my_model.fit(train_ds,
                                epochs=100,
                                validation_data=val_ds,
                                callbacks=[callback,lr_scheduler],
                                verbose=1)  # Set verbose=1 for detailed output


        loss, accuracy = my_model.evaluate(test_ds)

        print("Loss: ", loss)
        print("Accuracy: ", accuracy)










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
        serving_outputs = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=0))(serving_outputs)
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

 
        return tflite_file_path  # Return the path to the TFLite model and labels

    except Exception as e:
        print(f"Error creating TFLite model: {e}")
        return None, None  # Return None to indicate failure
    
if __name__ == '__main__':
    # Example usage:
    esc50_csv_path = 'local_folders\metadata.csv'
    base_data_path = ''

    tflite_model_path = create_tflite_model_from_csv(esc50_csv_path, base_data_path,"group_model")
    if tflite_model_path:
        print(f"TFLite model saved successfully at: {tflite_model_path}")
    else:
        print("Failed to create TFLite model.")
