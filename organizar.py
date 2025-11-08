import os
import shutil
import random
from tqdm import tqdm

# --- CONFIGURACION ---

# Directorio de las imagenes y etiquetas originales
DIRECTORIO_FUENTE_IMAGENES = 'yolov4/images'
DIRECTORIO_FUENTE_LABELS = 'yolov4/labels/train'

# Directorio donde se creara la nueva estructura 'train' y 'val'
DIRECTORIO_SALIDA = 'dataset_yolov4'

# Porcentaje de imagenes para el conjunto de entrenamiento (el resto será para validacion)
PORCENTAJE_TRAIN = 0.8

def dividir_dataset(img_source, lbl_source, output_dir, train_ratio):
    """
    Toma imagenes y etiquetas de un directorio fuente y las divide
    en subdirectorios de entrenamiento y validacion.
    """
    # Crear la estructura de carpetas de YOLO
    for split in ['train', 'val']:
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)

    # Obtener lista de imagenes y mezclarla
    try:
        image_files = sorted([f for f in os.listdir(img_source) if f.endswith(('.jpg', '.png'))])
    except FileNotFoundError:
        print(f"Error: No se encontro el directorio de imagenes fuente '{img_source}'")
        return
        
    random.shuffle(image_files)
    
    # Dividir la lista en entrenamiento y validacion
    split_point = int(len(image_files) * train_ratio)
    train_files = image_files[:split_point]
    val_files = image_files[split_point:]

    # Funcion para copiar archivos
    def copiar_archivos(file_list, split_name):
        img_dest = os.path.join(output_dir, 'images', split_name)
        lbl_dest = os.path.join(output_dir, 'labels', split_name)
        
        for filename in tqdm(file_list, desc=f"Copiando a '{split_name}'"):
            # Copiar imagen
            shutil.copy(os.path.join(img_source, filename), os.path.join(img_dest, filename))
            
            # Copiar etiqueta (si existe)
            label_filename = os.path.splitext(filename)[0] + '.txt'
            label_path_origen = os.path.join(lbl_source, label_filename)
            if os.path.exists(label_path_origen):
                shutil.copy(label_path_origen, os.path.join(lbl_dest, label_filename))
    
    # Copiar los archivos
    copiar_archivos(train_files, 'train')
    copiar_archivos(val_files, 'val')

    print(f"\nDataset dividido con exito en la carpeta '{output_dir}'")
    print(f"Entrenamiento: {len(train_files)} imagenes | Validacion: {len(val_files)} imagenes")

if __name__ == "__main__":
    dividir_dataset(DIRECTORIO_FUENTE_IMAGENES, DIRECTORIO_FUENTE_LABELS, DIRECTORIO_SALIDA, PORCENTAJE_TRAIN)