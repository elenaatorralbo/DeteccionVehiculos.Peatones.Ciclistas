import cv2
import numpy as np
import time
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from skimage.feature import hog
from skimage import color
from sklearn.decomposition import PCA

# --- CONFIGURACIÓN DE DESCRIPTORES ---
try:
    FEATURE_DETECTOR = cv2.SIFT_create(nfeatures=500)
except AttributeError:
    print("Advertencia: SIFT no encontrado. Usando detector FAST simple.")
    FEATURE_DETECTOR = cv2.FastFeatureDetector_create()

# --- CONFIGURACIÓN HOG ---
HOG_IMAGE_SIZE = (64, 64)
# --- CONFIGURACIÓN HISTOGRAMA HSV ---
HIST_BINS = 8  # 8 bins por canal (8*8*8 = 512 features)
# -------------------------

# --- CONFIGURACIÓN PCA ---
PCA_N_COMPONENTS = 0.95


# -------------------------


def calculate_hsv_histogram(img):
    """Calcula el histograma de 3D HSV (8x8x8) y lo normaliza."""
    # Se asegura que la imagen sea BGR antes de la conversión
    if img.ndim < 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Rango de Hue: 0-180, Saturación/Valor: 0-256
    hist = cv2.calcHist([img_hsv], [0, 1, 2], None,
                        [HIST_BINS, HIST_BINS, HIST_BINS],
                        [0, 180, 0, 256, 0, 256])

    # Aplanar y normalizar
    cv2.normalize(hist, hist)
    return hist.flatten()


def extract_features(cropped_data, set_name):
    """
    Calcula un vector de características (features) para cada vehículo recortado,
    incluyendo HOG y Histograma HSV.
    """

    X_features = []
    Y_labels = []

    for item in tqdm(cropped_data, desc=f"Extrayendo features de {set_name}"):
        img = item['image']
        obj_class = item['class']

        if img is None or img.size == 0: continue

        h, w = img.shape[:2]
        aspect_ratio = w / h

        # 1. Preparación para HOG
        try:
            resized_img_hog = cv2.resize(img, HOG_IMAGE_SIZE)
            gray_img = cv2.cvtColor(resized_img_hog, cv2.COLOR_BGR2GRAY)
        except cv2.error:
            continue

        # 2. Descriptores de Textura/Patrón (Número de KeyPoints)
        keypoints, _ = FEATURE_DETECTOR.detectAndCompute(img, None)
        num_keypoints = len(keypoints)

        # 3. Descriptores HOG
        hog_features = hog(gray_img, orientations=9, pixels_per_cell=(8, 8),
                           cells_per_block=(2, 2), block_norm='L2-Hys',
                           visualize=False, feature_vector=True)

        # 4. Histograma de Color (NUEVO)
        hsv_hist_features = calculate_hsv_histogram(img)

        # Vector de características: [Aspect Ratio, Num Keypoints] + [HOG Features] + [HSV Hist Features]
        base_features = [aspect_ratio, num_keypoints]
        feature_vector = np.concatenate((base_features, hog_features, hsv_hist_features))

        X_features.append(feature_vector)

        # Etiquetado Multiclase
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


def run_pca(X_train_scaled, X_test_scaled):
    """Aplica PCA para reducir la dimensionalidad."""
    pca = PCA(n_components=PCA_N_COMPONENTS, random_state=42)

    # Ajustar PCA solo en el conjunto de entrenamiento
    X_train_pca = pca.fit_transform(X_train_scaled)
    # Aplicar la misma transformación al conjunto de prueba
    X_test_pca = pca.transform(X_test_scaled)

    print(f"Dimensionalidad original: {X_train_scaled.shape[1]}")
    print(
        f"Dimensionalidad reducida (PCA): {X_train_pca.shape[1]} (Componentes que retienen el {PCA_N_COMPONENTS * 100}%)")

    return X_train_pca, X_test_pca, pca


# --- FUNCIÓN PRINCIPAL PARA IMPORTAR ---
def run_feature_extraction(train_set, test_set):
    start_time = time.time()

    # 1. Extracción
    X_train, Y_train, train_data_visual = extract_features(train_set, "Train")
    X_test, Y_test, test_data_visual = extract_features(test_set, "Test")

    # 2. Normalización
    X_train_scaled, X_test_scaled, scaler = normalize_features(X_train, X_test)

    # 3. PCA
    X_train_pca, X_test_pca, pca = run_pca(X_train_scaled, X_test_scaled)

    elapsed = time.time() - start_time
    minutes, seconds = divmod(elapsed, 60)

    print("\n=======================================================")
    print(f"FASE 3: EXTRACCIÓN (HOG + Histograma HSV), NORMALIZACIÓN Y PCA COMPLETA.")
    print(f"X_train (PCA) shape: {X_train_pca.shape}")
    print(f"TIEMPO TOTAL: {int(minutes)} minutos y {seconds:.2f} segundos.")
    print(f"=======================================================")

    return X_train_pca, Y_train, X_test_pca, Y_test, test_data_visual, scaler, pca