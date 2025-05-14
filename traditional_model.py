import os
import time
import pandas as pd
import numpy as np
import librosa

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# -------------------------------
# CONFIG & PATHS
# -------------------------------
base_path = os.path.join("environmental-sound-classification-50", "versions", "15")
data_path = os.path.join(base_path, "audio", "audio", "16000")
csv_path = os.path.join(base_path, "esc50.csv")
log_csv_path = "model_logs.csv"
log_txt_path = "model_logs.txt"

# -------------------------------
# LOGGING FUNCTION
# -------------------------------
def log_output(text):
    print(text)
    with open(log_txt_path, "a", encoding="utf-8") as f:
        f.write(text + "\n")

def log_model_results(model_name, best_params, cv_acc, test_acc, report):
    # Log to CSV
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

    # Log to TXT
    log_output(f"\n📌 Model: {model_name}")
    log_output(f"🎯 Best Parameters: {best_params}")
    log_output(f"📊 Cross-Validation Accuracy: {cv_acc:.4f}")
    log_output(f"✅ Test Accuracy: {test_acc:.4f}")
    log_output("📋 Classification Report:\n" + report)

# -------------------------------
# LOAD & PREPARE DATA
# -------------------------------
df = pd.read_csv(csv_path)

# Map to main categories
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
features, labels = [], []
for _, row in df.iterrows():
    file_path = os.path.join(data_path, row['filename'])
    y, sr = librosa.load(file_path, sr=16000, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    features.append(mfcc_mean)
    labels.append(row['main_category'])

X = np.array(features)
y = np.array(labels)

# -------------------------------
# ENCODE LABELS
# -------------------------------
encoder = OneHotEncoder(sparse_output=False)
y_onehot = encoder.fit_transform(y.reshape(-1, 1))
label_names = encoder.categories_[0]
y_labels = np.array([label_names[i] for i in np.argmax(y_onehot, axis=1)])

# -------------------------------
# TRAIN/TEST SPLIT
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y_labels, test_size=0.2, random_state=42, stratify=y_labels)

# -------------------------------
# RANDOM FOREST
# -------------------------------
rf_params = {
    'n_estimators': [50, 100],
    'max_depth': [10, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
}
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=5, scoring='accuracy', n_jobs=-1)
rf_grid.fit(X_train, y_train)
rf_preds = rf_grid.best_estimator_.predict(X_test)
log_model_results("RandomForest", rf_grid.best_params_, rf_grid.best_score_, accuracy_score(y_test, rf_preds), classification_report(y_test, rf_preds))

# -------------------------------
# SVM
# -------------------------------
svm_params = {
    'kernel': ['linear', 'rbf'],
    'C': [1, 10],
    'gamma': ['scale', 0.1],
}
start = time.time()
svm_grid = GridSearchCV(SVC(), svm_params, cv=5, scoring='accuracy', n_jobs=-1)
svm_grid.fit(X_train, y_train)
svm_preds = svm_grid.best_estimator_.predict(X_test)
log_output(f"\n⏱️ SVM Grid Search Time: {time.time() - start:.2f} seconds")
log_model_results("SVM", svm_grid.best_params_, svm_grid.best_score_, accuracy_score(y_test, svm_preds), classification_report(y_test, svm_preds))

# -------------------------------
# KNN
# -------------------------------
knn_params = {
    'n_neighbors': [3, 5],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'kd_tree'],
}
knn_grid = GridSearchCV(KNeighborsClassifier(), knn_params, cv=5, scoring='accuracy', n_jobs=-1)
knn_grid.fit(X_train, y_train)
knn_preds = knn_grid.best_estimator_.predict(X_test)
log_model_results("KNN", knn_grid.best_params_, knn_grid.best_score_, accuracy_score(y_test, knn_preds), classification_report(y_test, knn_preds))
