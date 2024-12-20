import os
import csv

# Map emotion codes to human-readable labels
EMOTION_MAP = {
    "01": "Neutral",
    "02": "Calm",
    "03": "Happy",
    "04": "Sad",
    "05": "Angry",
    "06": "Fearful",
    "07": "Disgust",
    "08": "Surprised"
}

def extract_labels(data_dir, output_file):
    """Generate a labels.csv file from the filenames in the RAVDESS dataset."""
    if not os.path.exists(data_dir):
        raise ValueError(f"The directory {data_dir} does not exist.")
    
    # Collect all .wav files
    files = [f for f in os.listdir(data_dir) if f.endswith(".wav")]
    
    with open(output_file, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(["filename", "emotion"])
        
        for file in files:
            try:
                # Extract emotion code from filename
                emotion_code = file.split("-")[2]
                emotion_label = EMOTION_MAP[emotion_code]
                writer.writerow([file, emotion_label])
            except KeyError:
                print(f"Skipping file {file}: Invalid emotion code.")
            except Exception as e:
                print(f"Error processing file {file}: {e}")
    
    print(f"Labels saved to {output_file}")
