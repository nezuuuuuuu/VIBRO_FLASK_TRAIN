import os
from pydub import AudioSegment
# from pydub.silence import detect_leading_silence # This line is commented out in your original script, keep it that way if not needed.

def trim_audio_files(input_folder, output_folder, trim_duration_ms=5000):
    """
    Trims audio files in the input_folder to a specified duration (default 5 seconds)
    and saves them to the output_folder.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    for filename in os.listdir(input_folder):
        # Ensure only common audio file extensions are processed
        if filename.lower().endswith(('.mp3', '.wav', '.flac', '.aac', '.ogg')):
            input_filepath = os.path.join(input_folder, filename)
            
            # Determine the file extension to use for the output format
            # Split the filename to get the base name and extension
            base_name, ext = os.path.splitext(filename)
            output_format = ext[1:].lower() # Get extension without the dot and convert to lowercase

            output_filepath = os.path.join(output_folder, filename)

            try:
                audio = AudioSegment.from_file(input_filepath)

                # Trim to the desired duration
                trimmed_audio = audio[:trim_duration_ms]

                # Export the trimmed audio to the output folder
                # Explicitly specify the format using the extracted extension
                trimmed_audio.export(output_filepath, format=output_format)
                print(f"Successfully trimmed and saved: {filename}")

            except Exception as e:
                print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    # Define your input and output folders
    input_folder_name = "speech"
    output_folder_name = "speech_trimmed"

    trim_audio_files(input_folder_name, output_folder_name, trim_duration_ms=5000)
    print("\nAudio trimming complete!")