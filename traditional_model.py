import os
import time
import numpy as np
import pandas as pd
import librosa
import warnings
import concurrent.futures

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

# -------------------------------
# CONFIG & PATHS
# -------------------------------
base_path = os.path.join("environmental-sound-classification-50", "versions", "15")
data_path = os.path.join(base_path, "audio", "audio", "16000")
csv_path = os.path.join(base_path, "esc50.csv")
log_csv_path = "model_logs.csv"
log_txt_path = "model_logs.txt"
features_cache = "features.npy"
labels_cache = "labels.npy"

# -------------------------------
# LOGGING
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
    log_output(f"📊 CV Accuracy: {cv_acc:.4f}")
    log_output(f"✅ Test Accuracy: {test_acc:.4f}")
    log_output("📋 Classification Report:\n" + report)

# -------------------------------
# LOAD & MAP DATA
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
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y).T, axis=0)
    rms = np.mean(librosa.feature.rms(y=y).T, axis=0)
    return np.hstack([mfcc, chroma, contrast, zcr, rms])

if os.path.exists(features_cache) and os.path.exists(labels_cache):
    log_output("✅ Loading cached features...")
    X = np.load(features_cache)
    y = np.load(labels_cache, allow_pickle=True)
else:
    log_output("🔄 Extracting features in parallel...")
    def process_row(row):
        file_path = os.path.join(data_path, row['filename'])
        try:
            return extract_features(file_path), row['main_category']
        except Exception as e:
            print(f"❌ Error processing {row['filename']}: {e}")
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(process_row, [row for _, row in df.iterrows()]))

    features, labels = zip(*[r for r in results if r is not None])
    X = np.array(features)
    y = np.array(labels)

    np.save(features_cache, X)
    np.save(labels_cache, y)
    log_output("💾 Features cached.")

# -------------------------------
# ENCODE, SPLIT, SCALE
# -------------------------------
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------
# MODEL TRAINING & LOGGING
# -------------------------------

# XGBoost
xgb_params = {'n_estimators': [100], 'max_depth': [4, 6], 'learning_rate': [0.1, 0.3], 'subsample': [0.8]}
xgb_grid = GridSearchCV(XGBClassifier(use_label_encoder=False, eval_metric='mlogloss',
                                      tree_method='gpu_hist', gpu_id=0, random_state=42),
                        xgb_params, cv=5, scoring='accuracy', n_jobs=-1)
xgb_grid.fit(X_train, y_train)
xgb_preds = xgb_grid.best_estimator_.predict(X_test)
xgb_report = classification_report(label_encoder.inverse_transform(y_test),
                                   label_encoder.inverse_transform(xgb_preds))
log_model_results("XGBoost (GPU)", xgb_grid.best_params_, xgb_grid.best_score_,
                  accuracy_score(y_test, xgb_preds), xgb_report)

# LightGBM
lgbm_params = {'n_estimators': [100], 'max_depth': [4, 6], 'learning_rate': [0.1, 0.3], 'num_leaves': [31, 50]}
lgbm_grid = GridSearchCV(LGBMClassifier(device='gpu', random_state=42),
                         lgbm_params, cv=5, scoring='accuracy', n_jobs=-1)
lgbm_grid.fit(X_train, y_train)
lgbm_preds = lgbm_grid.best_estimator_.predict(X_test)
lgbm_report = classification_report(label_encoder.inverse_transform(y_test),
                                    label_encoder.inverse_transform(lgbm_preds))
log_model_results("LightGBM (GPU)", lgbm_grid.best_params_, lgbm_grid.best_score_,
                  accuracy_score(y_test, lgbm_preds), lgbm_report)

# Random Forest
rf_params = {'n_estimators': [50, 100], 'max_depth': [10, None], 'min_samples_split': [2, 5], 'min_samples_leaf': [1, 2]}
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42),
                       rf_params, cv=5, scoring='accuracy', n_jobs=-1)
rf_grid.fit(X_train, y_train)
rf_preds = rf_grid.best_estimator_.predict(X_test)
rf_report = classification_report(label_encoder.inverse_transform(y_test),
                                  label_encoder.inverse_transform(rf_preds))
log_model_results("RandomForest", rf_grid.best_params_, rf_grid.best_score_,
                  accuracy_score(y_test, rf_preds), rf_report)

# SVM
svm_params = {'kernel': ['linear', 'rbf'], 'C': [1, 10], 'gamma': ['scale', 0.1]}
start = time.time()
svm_grid = GridSearchCV(SVC(), svm_params, cv=5, scoring='accuracy', n_jobs=-1)
svm_grid.fit(X_train, y_train)
svm_preds = svm_grid.best_estimator_.predict(X_test)
svm_report = classification_report(label_encoder.inverse_transform(y_test),
                                   label_encoder.inverse_transform(svm_preds))
log_output(f"\n⏱️ SVM Grid Search Time: {time.time() - start:.2f} seconds")
log_model_results("SVM", svm_grid.best_params_, svm_grid.best_score_,
                  accuracy_score(y_test, svm_preds), svm_report)

# KNN
knn_params = {'n_neighbors': [3, 5], 'weights': ['uniform', 'distance'], 'algorithm': ['auto', 'kd_tree']}
knn_grid = GridSearchCV(KNeighborsClassifier(), knn_params, cv=5, scoring='accuracy', n_jobs=-1)
knn_grid.fit(X_train, y_train)
knn_preds = knn_grid.best_estimator_.predict(X_test)
knn_report = classification_report(label_encoder.inverse_transform(y_test),
                                   label_encoder.inverse_transform(knn_preds))
log_model_results("KNN", knn_grid.best_params_, knn_grid.best_score_,
                  accuracy_score(y_test, knn_preds), knn_report)
