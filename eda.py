import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import librosa

# Print library versions
print("Library Versions:")
print("Librosa:", librosa.__version__)
print("Pandas:", pd.__version__)
print("NumPy:", np.__version__)
print("Seaborn:", sns.__version__)

# Set path
PATH = "environmental-sound-classification-50/versions/15/audio/audio/16000/"

# File inspection
files = os.listdir(PATH)
files_count = len(files)
print("\nFiles:")
if files:
    for file in files:
        print(file)
else:
    print("No files found.")

print(f"\nNumber of files: {files_count}")

# Load CSV metadata
df = pd.read_csv('environmental-sound-classification-50/versions/15/esc50.csv')
print("\nCSV Preview:")
print(df.head())
print(df.info())

print(f"\nTotal samples: {len(df)}")
print(f"Unique classes: {df['category'].nunique()}")
print("\nClass distribution:")
print(df['category'].value_counts())

# Check .wav file existence
audio_path = 'environmental-sound-classification-50/versions/15/audio/audio/16000/'
wav_files = os.listdir(audio_path)
print(f"\nNumber of .wav files: {len(wav_files)}")

csv_files = set(df['filename'])
missing = [f for f in wav_files if f not in csv_files]
print(f"Missing metadata entries: {len(missing)}")

# Extract duration and sample rate from first 100 files
durations = []
sample_rates = []

for f in df['filename'].head(100):
    y, sr = librosa.load(os.path.join(audio_path, f), sr=None)
    durations.append(len(y) / sr)
    sample_rates.append(sr)

print(f"\nAverage duration: {sum(durations)/len(durations):.2f} seconds")
print(f"Sample rates: {set(sample_rates)}")

# Plot class distribution
plt.figure(figsize=(12, 6))
sns.countplot(data=df, y='category', order=df['category'].value_counts().index)
plt.title("Class Distribution in ESC-50")
plt.tight_layout()
plt.savefig("class_distribution.png")
plt.show()

# Plot duration histogram
plt.hist(durations, bins=20, edgecolor='black')
plt.title("Distribution of Audio Durations (First 100 Samples)")
plt.xlabel("Duration (seconds)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("audio_durations_distribution.png")
plt.show()

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

df['main_category'] = df['category'].map(main_category_map)

main_category_distribution = df['main_category'].value_counts()

plt.figure(figsize=(10, 6))
main_category_distribution.plot(kind='bar', color='skyblue')
plt.title('ESC-50 Main Class Distribution')
plt.xlabel('Main Class')
plt.ylabel('Number of Samples')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig("main_class_distribution.png")
plt.show()

# ========================================
# PySpark Section: Main Category Distribution
# ========================================
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# Start Spark session
spark = SparkSession.builder \
    .appName("ESC50 Main Category Distribution") \
    .getOrCreate()

# Load CSV using Spark
esc50_sdf = spark.read.csv("environmental-sound-classification-50/versions/15/esc50.csv", header=True, inferSchema=True)

# Create mapping DataFrame for join
map_df = spark.createDataFrame(list(main_category_map.items()), ["category", "main_category"])

# Join and group by
df_with_main = esc50_sdf.join(map_df, on='category', how='left')

main_category_counts = df_with_main.groupBy("main_category").count().orderBy("count", ascending=False)

print("\nMain Category Distribution (Spark):")
main_category_counts.show(truncate=False)

# Optional: stop the Spark session
spark.stop()
