import os
import time
import pandas as pd
import numpy as np
import librosa
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# -------------------------------
# CONFIG & PATHS
# -------------------------------
base_path = os.path.join("environmental-sound-classification-50", "versions", "15")
data_path = os.path.join(base_path, "audio", "audio", "16000")
csv_path = os.path.join(base_path, "esc50.csv")
log_csv_path = "model_logs.csv"
log_txt_path = "model_logs.txt"

# -------------------------------
# LOGGING FUNCTIONS
# -------------------------------
def log_output(text):
    print(text)
    with open(log_txt_path, "a", encoding="utf-8") as f:
        f.write(text + "\n")

def log_model_results(model_name, best_params, cv_acc, test_acc, report):
    log_df = pd.DataFrame([{
        'model': model_name,
        'best_params': str(best_params),
        'cv_accuracy': cv_acc,
        'test_accuracy': test_acc,
        'classification_report': report.replace('\n', ' | ')
    }])
    if os.path.exists(log_csv_path):
        log_df.to_csv(log_csv_path, mode='a', header=False, index=False)
    else:
        log_df.to_csv(log_csv_path, index=False)

    log_output(f"\n📌 Model: {model_name}")
    log_output(f"🎯 Best Parameters: {best_params}")
    log_output(f"📊 Cross-Validation Accuracy: {cv_acc:.4f}")
    log_output(f"✅ Test Accuracy: {test_acc:.4f}")
    log_output("📋 Classification Report:\n" + report)

# -------------------------------
# LOAD & PREPARE DATA
# -------------------------------
df = pd.read_csv(csv_path)

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

# -------------------------------
# FEATURE EXTRACTION
# -------------------------------
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=16000, mono=True)

    # MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc.T, axis=0)

    # Chroma
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)

    # Spectral contrast
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)

    # Zero Crossing Rate
    zcr = np.mean(librosa.feature.zero_crossing_rate(y).T, axis=0)

    # RMS energy
    rms = np.mean(librosa.feature.rms(y=y).T, axis=0)

    return np.hstack([mfcc_mean, chroma, contrast, zcr, rms])

features, labels = [], []
for _, row in df.iterrows():
    file_path = os.path.join(data_path, row['filename'])
    try:
        feat = extract_features(file_path)
        features.append(feat)
        labels.append(row['main_category'])
    except Exception as e:
        print(f"Error processing {row['filename']}: {e}")

X = np.array(features)
y = np.array(labels)

# -------------------------------
# ENCODE LABELS
# -------------------------------
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# -------------------------------
# SPLIT AND SCALE
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------
# XGBOOST
# -------------------------------
xgb_params = {
    'n_estimators': [100],
    'max_depth': [4, 6],
    'learning_rate': [0.1, 0.3],
    'subsample': [0.8],
}
xgb_grid = GridSearchCV(
    XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
    xgb_params, cv=5, scoring='accuracy', n_jobs=-1
)
xgb_grid.fit(X_train, y_train)
xgb_preds = xgb_grid.best_estimator_.predict(X_test)

xgb_report = classification_report(
    label_encoder.inverse_transform(y_test),
    label_encoder.inverse_transform(xgb_preds)
)
log_model_results("XGBoost", xgb_grid.best_params_, xgb_grid.best_score_, accuracy_score(y_test, xgb_preds), xgb_report)

# -------------------------------
# LIGHTGBM
# -------------------------------
lgbm_params = {
    'n_estimators': [100],
    'max_depth': [4, 6],
    'learning_rate': [0.1, 0.3],
    'num_leaves': [31, 50],
}
lgbm_grid = GridSearchCV(
    LGBMClassifier(random_state=42),
    lgbm_params, cv=5, scoring='accuracy', n_jobs=-1
)
lgbm_grid.fit(X_train, y_train)
lgbm_preds = lgbm_grid.best_estimator_.predict(X_test)

lgbm_report = classification_report(
    label_encoder.inverse_transform(y_test),
    label_encoder.inverse_transform(lgbm_preds)
)
log_model_results("LightGBM", lgbm_grid.best_params_, lgbm_grid.best_score_, accuracy_score(y_test, lgbm_preds), lgbm_report)
