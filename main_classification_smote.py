import cv2
import numpy as np
import pandas as pd
import time
import os
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from skimage.feature import hog
# --- NUEVA IMPORTACIÓN PARA SMOTE ---
from imblearn.over_sampling import SMOTE
# -----------------------------------

# IMPORTAMOS LAS FASES ANTERIORES
from preprocess import run_preprocessing
# Importamos la NUEVA fase 3 y la función de histograma
from feature_extraction_v3 import run_feature_extraction, HOG_IMAGE_SIZE, calculate_hsv_histogram

# --- CONFIGURACIÓN ---
BASE_IMAGE_DIR = 'data_object_image_2/training/image_2/'
CLASS_ID_TO_NAME = {0: 'Car', 1: 'Truck', 2: 'Pedestrian', 3: 'Cyclist'}


# =========================================================================
# UTILITY: Función para guardar métricas
# =========================================================================

def save_metrics_to_file(model_name, accuracy, report):
    """Guarda el reporte de clasificación en un archivo .txt."""
    filename = f"metrics_{model_name}.txt"
    with open(filename, 'w') as f:
        f.write(f"--- REPORTE DE CLASIFICACIÓN: {model_name} ---\n\n")
        f.write(f"Precisión General (Accuracy): {accuracy:.4f}\n\n")
        f.write("Reporte Detallado:\n")
        f.write(report)
    print(f"\n✅ Métricas de {model_name} guardadas en {filename}")


# =========================================================================
# UTILITY: Función para clasificar un objeto individual (HOG+HSV Hist+PCA)
# =========================================================================

def classify_single_object(cropped_img, scaler, pca_model, svc_model):
    """Extrae features (HOG, HSV Hist incluido), normaliza, aplica PCA y clasifica con SVC."""

    h, w = cropped_img.shape[:2]
    aspect_ratio = w / h

    if h <= 0 or w <= 0: return "N/A"

    # --- HOG: Redimensionamiento y escala de grises ---
    try:
        resized_img_hog = cv2.resize(cropped_img, HOG_IMAGE_SIZE)
        gray_img = cv2.cvtColor(resized_img_hog, cv2.COLOR_BGR2GRAY)
    except cv2.error:
        return "N/A"
    # --------------------------------------------------

    # 1. Descriptores de Textura/Patrón (Número de KeyPoints)
    try:
        detector = cv2.SIFT_create(nfeatures=500)
    except AttributeError:
        detector = cv2.FastFeatureDetector_create()

    keypoints, _ = detector.detectAndCompute(cropped_img, None)
    num_keypoints = len(keypoints)

    # 2. Cálculo de HOG
    hog_features = hog(gray_img, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), block_norm='L2-Hys',
                       visualize=False, feature_vector=True)

    # 3. Histograma de Color (NUEVO)
    hsv_hist_features = calculate_hsv_histogram(cropped_img)

    # 4. Vector de Features (Base + HOG + Hist)
    base_features = np.array([aspect_ratio, num_keypoints])
    single_feature_vector = np.concatenate((base_features, hog_features, hsv_hist_features)).reshape(1, -1)

    # 5. Normalización
    single_scaled_feature = scaler.transform(single_feature_vector)

    # 6. Aplicar PCA
    single_pca_feature = pca_model.transform(single_scaled_feature)

    # 7. Predicción con SVC
    prediction = svc_model.predict(single_pca_feature)[0]

    # 8. Interpretación
    predicted_class = CLASS_ID_TO_NAME.get(prediction, "N/A")
    return predicted_class


# =========================================================================
# FUNCIÓN DE CLASIFICACIÓN Y ENTRENAMIENTO (FASE 4)
# =========================================================================

def classify_vehicles(X_train_pca, Y_train, X_test_pca, Y_test, test_data_visual):
    print("\n=======================================================")
    print("FASE 4: CLASIFICACIÓN Y PREDICCIÓN (SVC + SMOTE)")
    print("=======================================================")

    # --- PASO CRÍTICO: SOBREMUESTREO CON SMOTE ---
    start_smote = time.time()
    print("Aplicando SMOTE para balancear las clases en el conjunto de entrenamiento...")

    # SMOTE solo se aplica a los datos de entrenamiento para generar muestras sintéticas
    sm = SMOTE(random_state=42)
    X_train_res, Y_train_res = sm.fit_resample(X_train_pca, Y_train)

    elapsed_smote = time.time() - start_smote
    print(f"SMOTE completado en {elapsed_smote:.2f} segundos.")
    print(f"X_train original shape: {X_train_pca.shape} -> X_train resampled shape: {X_train_res.shape}")
    # ---------------------------------------------

    # 1. CLASIFICACIÓN SUPERVISADA: SVC
    start_svc = time.time()

    # IMPORTANTE: Eliminamos class_weight='balanced' porque los datos ya están balanceados con SMOTE
    svc_model = SVC(kernel='rbf', random_state=42)

    # Entrenar el modelo con las características BALANCEADAS
    svc_model.fit(X_train_res, Y_train_res)

    Y_pred_svc = svc_model.predict(X_test_pca)
    elapsed_svc = time.time() - start_svc

    print(f"\n--- SVC (kernel='rbf') ---")
    print(f"Tiempo de entrenamiento y predicción: {elapsed_svc:.4f} segundos.")

    # --- CÁLCULO DE MÉTRICAS DE RENDIMIENTO ---
    svc_accuracy = accuracy_score(Y_test, Y_pred_svc)
    target_names = list(CLASS_ID_TO_NAME.values())
    svc_report = classification_report(Y_test, Y_pred_svc, target_names=target_names)

    print(f"\nPrecisión General (Accuracy): {svc_accuracy:.4f} ({svc_accuracy * 100:.2f}%)")
    print("\nReporte de Clasificación (SVC + SMOTE - Precisión, Recall, F1-Score):")
    print(svc_report)

    # GUARDAR MÉTRICAS
    save_metrics_to_file("svc_smote", svc_accuracy, svc_report)
    # ----------------------------------------

    # 2. AGRUPAMIENTO NO SUPERVISADO: K-MEANS
    start_kmeans = time.time()
    # K-Means sigue usando los datos con PCA (no resampleados)
    kmeans_model = KMeans(n_clusters=4, random_state=42, n_init=10)
    kmeans_model.fit(X_train_pca)
    Y_pred_kmeans = kmeans_model.predict(X_test_pca)
    elapsed_kmeans = time.time() - start_kmeans

    print(f"\n--- K-MEANS (K=4 sobre datos PCA) ---")
    print(f"Tiempo de entrenamiento y predicción: {elapsed_kmeans:.4f} segundos.")

    # GENERACIÓN DE RESULTADOS
    results_df = pd.DataFrame({
        'Ground_Truth': Y_test,
        'Pred_SVC': Y_pred_svc,
        'Pred_KMeans': Y_pred_kmeans,
        'Image_Data': test_data_visual
    })

    map_to_class_name = lambda x: CLASS_ID_TO_NAME.get(x, 'Desconocido')

    results_df['GT_Class'] = results_df['Ground_Truth'].apply(map_to_class_name)
    results_df['SVC_Class'] = results_df['Pred_SVC'].apply(map_to_class_name)
    results_df['KMeans_Cluster'] = results_df['Pred_KMeans'].apply(lambda x: f'Cluster {x}')

    print("\n--- RESULTADOS (Primeras 10 predicciones) ---")
    print(results_df[['GT_Class', 'SVC_Class', 'KMeans_Cluster']].head(10))

    return results_df, svc_model


# =========================================================================
# FUNCIONES DE VISUALIZACIÓN REFORZADAS
# =========================================================================

def inspect_predictions(results_df, num_samples=5):
    """Muestra visualmente algunas predicciones (solo recortes) del Test Set."""

    print(f"\n--- INSPECCIÓN VISUAL DE RECORTES ({num_samples} MUESTRAS) ---")
    print("Ventana de OpenCV: Presiona cualquier tecla para pasar al siguiente recorte.")

    cv2.waitKey(1)

    for i in range(min(num_samples, len(results_df))):
        row = results_df.iloc[i]

        img = row['Image_Data']['image']
        gt = row['GT_Class']
        pred = row['SVC_Class']

        color = (0, 255, 0) if (pred == gt) else (0, 0, 255)
        text_label = f"RECORTES PRED: {pred} (GT: {gt})"

        cv2.imshow(text_label, img)
        cv2.waitKey(0)

    cv2.destroyAllWindows()


def run_realtime_detection(svc_model, scaler, pca_model, results_df, num_samples=5):
    """
    Usa los Bounding Boxes (BBox) del GT para clasificar cada región con SVC + SMOTE.
    """

    print("\n--- INICIO DE LA INSPECCIÓN VISUAL EN IMÁGENES COMPLETAS ---")
    print(f"Mostrando {num_samples} predicciones del Test Set usando BBOX del Ground Truth.")
    print("Ventana de OpenCV: Presiona cualquier tecla para pasar a la siguiente imagen.")

    cv2.waitKey(1)

    for i in range(min(num_samples, len(results_df))):
        row = results_df.iloc[i]

        image_id = row['Image_Data']['image_id']
        bbox_float = row['Image_Data']['bbox']

        test_image_path = os.path.join(BASE_IMAGE_DIR, f'{image_id}.png')

        img = cv2.imread(test_image_path)
        if img is None:
            continue

        xmin, ymin, xmax, ymax = map(int, bbox_float)
        detected_roi = img[ymin:ymax, xmin:xmax]

        # Clasificar la región usando el modelo SVC y PCA (con lógica de features actualizada)
        predicted_class = classify_single_object(detected_roi, scaler, pca_model, svc_model)

        gt_class = row['GT_Class']

        is_correct = (predicted_class == gt_class)
        color = (0, 255, 0) if is_correct else (0, 0, 255)
        thickness = 3

        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, thickness)

        text_label = f"PRED: {predicted_class} (GT: {gt_class})"
        cv2.putText(img, text_label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)

        cv2.imshow(f"FINAL - ID: {image_id}", img)
        cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    # 1. FASE 1 & 2: Preprocesamiento y Recorte
    train_set, test_set = run_preprocessing()

    if train_set and test_set:
        # 2. FASE 3: Extracción (HOG + Histograma HSV), Normalización y PCA
        X_train_pca, Y_train, X_test_pca, Y_test, test_data_visual, scaler, pca = run_feature_extraction(train_set,
                                                                                                         test_set)

        # 3. FASE 4: Clasificación y Obtención de Resultados con SVC + SMOTE
        results_df, svc_model = classify_vehicles(X_train_pca, Y_train, X_test_pca, Y_test, test_data_visual)

        # 4. INSPECCIÓN VISUAL
        run_realtime_detection(svc_model, scaler, pca, results_df, num_samples=40)