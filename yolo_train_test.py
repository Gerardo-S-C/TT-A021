from ultralytics import YOLO
import torch

# --- CONFIGURACION ---
# Ruta a tu archivo de configuracion de CVAT
ARCHIVO_YAML = './dataset_yolov4/data.yaml'

# --- ENTRENAMIENTO ---
def entrenar():
    # Cargar un modelo YOLOv8
    # '.pt' es un archivo de "pesos" que contiene el conocimiento del modelo.
    # Al cargarlo, hacemos "Transfer Learning": transferimos el conocimiento de un
    # modelo experto para especializarlo en nuestra tarea.
    model = YOLO('runs/detect/yolo_v3/weights/best.pt')

    print("Modelo cargado. Iniciando entrenamiento...")
    
    # Entrenamos el modelo con nuestros datos
    results = model.train(
        data=ARCHIVO_YAML,       # Tu archivo de configuracion que le dice a YOLO donde estan los datos
        epochs=100,              # No. de epocas
        imgsz=(576, 736),        # Redimensionar las imagenes a multiplos de 32
        patience=10,             # Detiene el entrenamiento si no hay mejora despues de 10 epocas
        batch=10,                # Numero de imagenes a procesar a la vez
        name='yolo_v4',          # Nombre para la carpeta de resultados
        exist_ok=True            # Sobrescribe la carpeta de resultados si ya existe
    )

    print("¡Entrenamiento completado!")

if __name__ == '__main__':
    # Verificar si hay una GPU disponible para el entrenamiento
    print(f"CUDA disponible: {torch.cuda.is_available()}")
    entrenar()