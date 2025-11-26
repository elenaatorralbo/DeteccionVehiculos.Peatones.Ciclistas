import cv2
import numpy as np
import time
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# --- CONFIGURACIÓN DE DESCRIPTORES ---
try:
    # Usaremos SIFT con un límite de features para reducir la memoria
    FEATURE_DETECTOR = cv2.SIFT_create(nfeatures=500)
except AttributeError:
    print("Advertencia: SIFT no encontrado. Usando detector FAST simple.")
    FEATURE_DETECTOR = cv2.FastFeatureDetector_create()


def extract_features(cropped_data, set_name):
    """
    Calcula un vector de características (features) para cada vehículo recortado.
    Se actualiza el etiquetado para incluir Car (0), Truck (1), Pedestrian (2), y Cyclist (3).
    """

    X_features = []
    Y_labels = []

    for item in tqdm(cropped_data, desc=f"Extrayendo features de {set_name}"):
        img = item['image']
        obj_class = item['class']

        if img is None or img.size == 0: continue

        h, w = img.shape[:2]
        aspect_ratio = w / h

        # 2. Color Promedio (HSV)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h_avg = np.mean(img_hsv[:, :, 0])
        s_avg = np.mean(img_hsv[:, :, 1])
        v_avg = np.mean(img_hsv[:, :, 2])

        # 3. Descriptores de Textura/Patrón (Número de KeyPoints)
        keypoints, _ = FEATURE_DETECTOR.detectAndCompute(img, None)
        num_keypoints = len(keypoints)

        feature_vector = [aspect_ratio, h_avg, s_avg, v_avg, num_keypoints]

        X_features.append(feature_vector)

        # MODIFICACIÓN: Etiquetado Multiclase (Car=0, Truck=1, Pedestrian=2, Cyclist=3)
        if obj_class == 'Car':
            label = 0
        elif obj_class == 'Truck':
            label = 1
        elif obj_class == 'Pedestrian':
            label = 2
        elif obj_class == 'Cyclist':
            label = 3
        else:
            continue

        Y_labels.append(label)

    return np.array(X_features), np.array(Y_labels), cropped_data


def normalize_features(X_train, X_test):
    """Normaliza las características usando StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler


# --- FUNCIÓN PRINCIPAL PARA IMPORTAR ---
def run_feature_extraction(train_set, test_set):
    start_time = time.time()

    # 1. Extracción
    X_train, Y_train, train_data_visual = extract_features(train_set, "Train")
    X_test, Y_test, test_data_visual = extract_features(test_set, "Test")

    # 2. Normalización
    X_train_scaled, X_test_scaled, scaler = normalize_features(X_train, X_test)

    elapsed = time.time() - start_time
    minutes, seconds = divmod(elapsed, 60)

    print("\n=======================================================")
    print(f"FASE 3: EXTRACCIÓN Y NORMALIZACIÓN COMPLETA.")
    print(f"X_train (escalado) shape: {X_train_scaled.shape}")
    print(f"TIEMPO TOTAL: {int(minutes)} minutos y {seconds:.2f} segundos.")
    print(f"=======================================================")

    return X_train_scaled, Y_train, X_test_scaled, Y_test, test_data_visual, scaler