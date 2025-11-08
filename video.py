from ultralytics import YOLO
import cv2
import os
from tqdm import tqdm

# --- CONFIGURACIÓN ---
# 1. Ruta a tu modelo entrenado (el archivo best.pt)
RUTA_MODELO = 'runs/detect/yolo_v3/weights/best.pt'

# 2. Directorio donde están TODAS las imágenes a procesar
DIRECTORIO_FUENTE = 'AZCONT_preprocesadas'

# 3. Carpeta donde se guardarán los resultados (los frames del video)
DIRECTORIO_SALIDA = 'resultados_video_AZ_CONT'

# 4. Umbral de confianza
UMBRAL_CONFIANZA = 0.5

# --- PROCESAMIENTO EN LOTE ---

# Crear el directorio de salida si no existe
os.makedirs(DIRECTORIO_SALIDA, exist_ok=True)

# Cargar tu modelo personalizado
try:
    model = YOLO(RUTA_MODELO)
    nombres_clases = model.names
    print("Modelo cargado con éxito.")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    exit()

# Obtener la lista de imágenes y ordenarla para que tenga secuencia de video
try:
    lista_imagenes = sorted([f for f in os.listdir(DIRECTORIO_FUENTE) if f.endswith(('.jpg', '.png'))])
    if not lista_imagenes:
        print(f"Error: No se encontraron imágenes en '{DIRECTORIO_FUENTE}'")
        exit()
except FileNotFoundError:
    print(f"Error: El directorio '{DIRECTORIO_FUENTE}' no fue encontrado.")
    exit()

print(f"Procesando {len(lista_imagenes)} imágenes...")

# Iterar sobre cada imagen de la carpeta con una barra de progreso
for nombre_imagen in tqdm(lista_imagenes, desc="Procesando lote"):
    ruta_completa = os.path.join(DIRECTORIO_FUENTE, nombre_imagen)
    
    # Realizar la predicción en la imagen actual
    results = model.predict(source=ruta_completa, conf=UMBRAL_CONFIANZA, save=False, verbose=False)

    # Cargar la imagen original para dibujar sobre ella
    img = cv2.imread(ruta_completa)
    overlay = img.copy()

    contador_personas = 0
    hay_cigarro = False

    # El resultado de 'predict' es una lista, tomamos el primer elemento
    r = results[0]
    
    # Dibujar las cajas de cada detección
    boxes = r.boxes
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        
        nombre_clase = nombres_clases[cls]
        etiqueta = f'{nombre_clase} {conf:.2f}'
        color = (255, 0, 0) if nombre_clase == 'persona' else (0, 255, 255)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, etiqueta, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Contar los objetos
        if cls == 0:
            contador_personas += 1
        elif cls == 1:
            hay_cigarro = True

    # Preparar y dibujar el texto de resumen
    texto_personas = f"Personas: {contador_personas}"
    texto_cigarros = f"Cigarros: {'SI' if hay_cigarro else 'NO'}"
    
    cv2.rectangle(overlay, (10, 10), (250, 70), (0,0,0), -1)
    alpha = 0.6
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    cv2.putText(img, texto_personas, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
    cv2.putText(img, texto_cigarros, (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
    
    # Guardar el frame procesado en la carpeta de salida
    ruta_salida = os.path.join(DIRECTORIO_SALIDA, nombre_imagen)
    cv2.imwrite(ruta_salida, img)

print(f"\n¡Proceso completado! Todos los frames han sido guardados en la carpeta '{DIRECTORIO_SALIDA}'.")