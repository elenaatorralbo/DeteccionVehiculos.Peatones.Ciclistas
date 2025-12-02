import os
import cv2
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# --- CONFIGURACIÓN DE RUTAS Y RATIOS SOLICITADOS ---
BASE_IMAGE_DIR = 'data_object_image_2/training/image_2/'
BASE_LABEL_DIR = 'data_object_label_2/training/label_2'
TARGET_CLASSES = ['Car', 'Truck', 'Pedestrian', 'Cyclist']

# Parámetros para reducir el dataset y dividirlo
DATA_USAGE_RATIO = 0.70
TEST_SIZE = 0.20
RANDOM_SEED = 42


def parse_kitti_labels():
    """Lee y procesa todos los archivos de etiquetas de KITTI."""
    print("Fase 1: Leyendo y filtrando archivos de etiquetas...")
    all_annotations = []

    if not os.path.isdir(BASE_LABEL_DIR):
        print(f"Error de ruta: No se encuentra el directorio de etiquetas: {BASE_LABEL_DIR}")
        return pd.DataFrame()

    label_files = [f for f in os.listdir(BASE_LABEL_DIR) if f.endswith('.txt')]
    for filename in tqdm(label_files, desc="Leyendo etiquetas"):
        file_path = os.path.join(BASE_LABEL_DIR, filename)
        image_id = filename.replace('.txt', '')

        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts: continue

                obj_class = parts[0]

                if obj_class in TARGET_CLASSES:
                    try:
                        # BBox 2D: xmin, ymin, xmax, ymax (índices 4 a 7)
                        xmin = float(parts[4])
                        ymin = float(parts[5])
                        xmax = float(parts[6])
                        ymax = float(parts[7])

                        all_annotations.append({
                            'image_id': image_id,
                            'class': obj_class,
                            'bbox': [xmin, ymin, xmax, ymax],
                        })
                    except (ValueError, IndexError):
                        continue

    return pd.DataFrame(all_annotations)


def create_datasets_and_crop(df):
    """Aplica el ratio de uso, divide en Train/Test y realiza el recorte de imágenes."""

    print("\nFase 2: Dividiendo y preparando los conjuntos de datos...")

    # PASO 1: APLICAR RATIO DE USO
    all_image_ids = df['image_id'].unique()
    num_images_to_use = int(len(all_image_ids) * DATA_USAGE_RATIO)

    np.random.seed(RANDOM_SEED)
    sampled_image_ids = np.random.choice(
        all_image_ids,
        size=num_images_to_use,
        replace=False
    )

    df_sampled = df[df['image_id'].isin(sampled_image_ids)].reset_index(drop=True)

    # 2. DIVISIÓN INTERNA (80/20 del subconjunto)
    train_ids, test_ids = train_test_split(
        sampled_image_ids,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED
    )

    train_df = df_sampled[df_sampled['image_id'].isin(train_ids)].reset_index(drop=True)
    test_df = df_sampled[df_sampled['image_id'].isin(test_ids)].reset_index(drop=True)

    print(f"Total de Anotaciones filtradas inicialmente: {len(df)}")
    print(f"Imágenes únicas seleccionadas ({DATA_USAGE_RATIO * 100}%): {len(sampled_image_ids)}")
    print(f"Anotaciones finales para Train/Test: {len(df_sampled)}")
    print(f"Imágenes únicas para entrenamiento: {len(train_ids)}")
    print(f"Imágenes únicas para prueba: {len(test_ids)}")

    # 3. RECORTES (Añadimos tqdm)

    def crop_images_from_df(dataset_df, set_name):
        cropped_data = []

        for index, row in tqdm(dataset_df.iterrows(), total=len(dataset_df), desc=f"Recortando {set_name}"):
            image_filename = row['image_id'] + '.png'
            image_path = os.path.join(BASE_IMAGE_DIR, image_filename)

            img = cv2.imread(image_path)

            if img is None:
                continue

                # Coordenadas BBox (convertidas a entero)
            xmin, ymin, xmax, ymax = map(int, row['bbox'])

            # Recorte: [ymin:ymax, xmin:xmax]
            cropped_img = img[ymin:ymax, xmin:xmax]

            if cropped_img.size == 0 or cropped_img.shape[0] < 20 or cropped_img.shape[1] < 20:
                continue

            # Almacenar la imagen recortada, la clase y el ID
            cropped_data.append({
                'image': cropped_img,
                'class': row['class'],
                'image_id': row['image_id'],
                'bbox': row['bbox']
            })

        print(f"Recortes válidos generados para {set_name}: {len(cropped_data)}")
        return cropped_data

    train_data = crop_images_from_df(train_df, "Train")
    test_data = crop_images_from_df(test_df, "Test")

    return train_data, test_data


# --- FUNCIÓN PRINCIPAL PARA IMPORTAR ---
def run_preprocessing():
    start_time = time.time()

    kitti_df = parse_kitti_labels()
    if kitti_df.empty:
        elapsed = time.time() - start_time
        print(f"\n--- TIEMPO TOTAL: {elapsed:.2f} segundos ---")
        return [], []

    train_set, test_set = create_datasets_and_crop(kitti_df)

    elapsed = time.time() - start_time
    minutes, seconds = divmod(elapsed, 60)
    print(f"\n=======================================================")
    print(f"PREPROCESAMIENTO COMPLETO. FASE 2 LISTA.")
    print(f"TIEMPO TOTAL: {int(minutes)} minutos y {seconds:.2f} segundos.")
    print(f"=======================================================")

    return train_set, test_set


if __name__ == '__main__':
    # No se ejecuta automáticamente al importar
    train_set, test_set = run_preprocessing()
    if train_set:
        print("Preprocesamiento completado. Listo para la Fase 3.")