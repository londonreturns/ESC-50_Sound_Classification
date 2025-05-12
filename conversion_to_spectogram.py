import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# Paths
data = "environmental-sound-classification-50\\versions\\15\\audio\\audio\\16000\\"
csv_path = "environmental-sound-classification-50\\versions\\15\\esc50.csv"
output_root = "spectrogram"

# Load dataset CSV
df = pd.read_csv(csv_path)

# Map detailed classes to broader categories
main_category_map = {
    'dog': 'Animals', 'rooster': 'Animals', 'pig': 'Animals', 'cow': 'Animals', 'frog': 'Animals',
    'cat': 'Animals', 'hen': 'Animals', 'insects': 'Animals', 'sheep': 'Animals', 'crow': 'Animals',
    'rain': 'Natural soundscapes & water sounds', 'sea_waves': 'Natural soundscapes & water sounds',
    'crackling_fire': 'Natural soundscapes & water sounds', 'crickets': 'Natural soundscapes & water sounds',
    'chirping_birds': 'Natural soundscapes & water sounds', 'water_drops': 'Natural soundscapes & water sounds',
    'wind': 'Natural soundscapes & water sounds', 'pouring_water': 'Natural soundscapes & water sounds',
    'toilet_flush': 'Natural soundscapes & water sounds', 'thunderstorm': 'Natural soundscapes & water sounds',
    'crying_baby': 'Human, non-speech sounds', 'sneezing': 'Human, non-speech sounds',
    'clapping': 'Human, non-speech sounds', 'breathing': 'Human, non-speech sounds',
    'coughing': 'Human, non-speech sounds', 'footsteps': 'Human, non-speech sounds',
    'laughing': 'Human, non-speech sounds', 'brushing_teeth': 'Human, non-speech sounds',
    'snoring': 'Human, non-speech sounds', 'drinking_sipping': 'Human, non-speech sounds',
    'door_wood_knock': 'Interior/domestic sounds', 'mouse_click': 'Interior/domestic sounds',
    'keyboard_typing': 'Interior/domestic sounds', 'door_wood_creaks': 'Interior/domestic sounds',
    'can_opening': 'Interior/domestic sounds', 'washing_machine': 'Interior/domestic sounds',
    'vacuum_cleaner': 'Interior/domestic sounds', 'clock_alarm': 'Interior/domestic sounds',
    'clock_tick': 'Interior/domestic sounds', 'glass_breaking': 'Interior/domestic sounds',
    'helicopter': 'Exterior/urban noises', 'chainsaw': 'Exterior/urban noises',
    'siren': 'Exterior/urban noises', 'car_horn': 'Exterior/urban noises', 'engine': 'Exterior/urban noises',
    'train': 'Exterior/urban noises', 'church_bells': 'Exterior/urban noises',
    'airplane': 'Exterior/urban noises', 'fireworks': 'Exterior/urban noises', 'hand_saw': 'Exterior/urban noises'
}

# Add main category to dataframe
df['main_category'] = df['category'].map(main_category_map)

# Process each audio file
for idx, row in df.iterrows():
    filename = row['filename']
    label = row['category']
    main_class = row['main_category']

    if pd.isna(main_class):
        print(f"Skipping unknown category: {label}")
        continue

    # Create main class folder if it doesn't exist
    output_dir = os.path.join(output_root, main_class)
    os.makedirs(output_dir, exist_ok=True)

    # Load and process audio
    file_path = os.path.join(data, filename)
    try:
        y, sr = librosa.load(file_path)
        y_trimmed, _ = librosa.effects.trim(y)
        S = librosa.feature.melspectrogram(y=y_trimmed, sr=sr)
        S_db = librosa.amplitude_to_db(S, ref=np.max)

        # Plot and save spectrogram
        plt.figure(figsize=(5, 4))
        librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel')
        plt.axis('off')
        plt.tight_layout()

        output_filename = f"{os.path.splitext(filename)[0]}.png"
        output_path = os.path.join(output_dir, output_filename)
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        print(f"Saved: {output_path}")

    except Exception as e:
        print(f"Error processing {filename}: {e}")
