import cv2
import time
import os
import torch
import csv
from datetime import datetime
from ultralytics import YOLO

# --- CONFIGURACIÓN GENERAL ---
MODEL_PATH = r"C:/Users/isria/Documents/ESCOM/TT/best_1.pt"  # o "./modelo_exportado/best.onnx"
CAM_INDEX  = 1                             # 0 = cámara por defecto

# --- PARÁMETROS DE PREPROCESAMIENTO ---
# Dimensiones a las que se redimensionará el frame ANTES de pasarlo a YOLO
PREPROCESS_WIDTH = 736
PREPROCESS_HEIGHT = 576

# --- PARÁMETROS DE INFERENCIA ---
CONF  = 0.70  # Confianza mínima para una detección
IOU   = 0.30  # IoU para el algoritmo Non-Maximum Suppression (NMS)
IMGSZ = (576, 736)   # Tamaño de entrada del modelo

# --- CONFIGURACIÓN DE VIDEO (OPCIONAL) ---
# (Opcional) Grabar la salida a un archivo de video
SAVE_VIDEO = True
OUTPUT_VIDEO_PATH = "vela1persona.mp4"

# --- NUEVO: CONFIGURACIÓN DE ALMACENAMIENTO ---
CSV_DIR = "datos_csv"
CSV_PATH = os.path.join(CSV_DIR, "resultados.csv")
IMAGES_DIR = "imagenes"  # usar "imágenes" si deseas, pero cuidado con el acento en algunas consolas
ID_CAMARA_FIJO = "camara_UT-MN256"

# Crear carpetas si no existen
os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

# Inicializar CSV con encabezados si no existe
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Date", "Time", "IdCamara", "ruta_imagentermica.jpg", "NumPersonas", "BanderaFumando"])

# Contador incremental de filas (ID)
# Si el CSV ya tenía datos, continuamos el conteo
def _leer_ultimo_id(csv_path: str) -> int:
    try:
        with open(csv_path, mode="r", encoding="utf-8") as f:
            lines = f.readlines()
            if len(lines) <= 1:
                return 0
            # Última línea con datos
            last = lines[-1].strip().split(",")[0]
            return int(last)
    except Exception:
        return 0

ID_CONTADOR = _leer_ultimo_id(CSV_PATH)

# --- NUEVO: Función para guardar imagen con base en la fecha dd/mm/YY ---
def guardar_imagen_screenshot(frame_bgr):
    """
    Guarda un screenshot del frame anotado en ./imagenes
    Nombre base: fecha dd/mm/YY (para archivo se usa dd-mm-YY.jpg)
    Si existe, agrega sufijos _001, _002, ...
    Retorna la ruta absoluta/relativa guardada que se pondrá en el CSV.
    """
    # Fecha para archivo (sin '/')
    fecha_archivo = datetime.now().strftime("%d-%m-%y")  # dd-mm-YY
    base_name = f"{fecha_archivo}.jpg"
    output_path = os.path.join(IMAGES_DIR, base_name)

    # Evitar sobrescritura: si existe, agregar sufijo incremental
    if os.path.exists(output_path):
        suf = 1
        while True:
            candidate = os.path.join(IMAGES_DIR, f"{fecha_archivo}_{suf:03d}.jpg")
            if not os.path.exists(candidate):
                output_path = candidate
                break
            suf += 1

    cv2.imwrite(output_path, frame_bgr)
    return output_path

# --- NUEVO: Función para escribir una fila en el CSV ---
def guardar_fila_csv(id_val, date_str, time_str, id_camara, ruta_img, num_personas, bandera_fumando):
    with open(CSV_PATH, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([id_val, date_str, time_str, id_camara, ruta_img, num_personas, bandera_fumando])

# --------------------------------------------------------------------------
# 1. Cargar el modelo
assert os.path.exists(MODEL_PATH), f"Error: No se encontró el modelo en la ruta: {MODEL_PATH}"
model = YOLO(MODEL_PATH)

# 2. Seleccionar dispositivo de cómputo (GPU o CPU)
device = "cuda" if torch.cuda.is_available() and MODEL_PATH.endswith(".pt") else "cpu"
print(f"Usando modelo: '{MODEL_PATH}'")
print(f"Usando dispositivo: '{device}'")

# 3. Función auxiliar para inicializar la cámara
def iniciar_camara(cam_index):
    """Intenta abrir una cámara con el índice especificado"""
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        return None
    
    # Fijar una resolución de captura alta para la visualización
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, PREPROCESS_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, PREPROCESS_HEIGHT)
    
    # Obtener las dimensiones reales de la cámara
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Cámara {cam_index} abierta - Resolución: {w}x{h}")
    
    return cap, w, h

# Iniciar captura de video
resultado = iniciar_camara(CAM_INDEX)
if resultado is None:
    print(f"Error: No se pudo abrir la cámara con índice {CAM_INDEX}.")
    exit()
cap, original_w, original_h = resultado

# 4. (Opcional) Configurar el escritor de video si está activado
writer = None
if SAVE_VIDEO:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, 20.0, (original_w, original_h))
    print(f"La salida de video se guardará en: '{OUTPUT_VIDEO_PATH}'")

# Variables para cálculo de FPS
prev_t = time.time()
fps_smooth = None

# --- BUCLE PRINCIPAL DE PROCESAMIENTO ---
print("\n--- CONTROLES ---")
print("Presiona 'P' para cambiar de cámara")
print("Presiona 'Q' o ESC para salir\n")

while True:
    # Leer un fotograma de la cámara
    ok, frame_original = cap.read()
    if not ok:
        print(" No se pudo leer el fotograma de la cámara. Terminando...")
        break

    # --- INICIO DEL PREPROCESAMIENTO ---
    # 1. Convertir a escala de grises
    frame_gris = cv2.cvtColor(frame_original, cv2.COLOR_BGR2GRAY)
    # 2. Redimensionar a las dimensiones deseadas
    frame_preprocesado = cv2.resize(frame_gris, (PREPROCESS_WIDTH, PREPROCESS_HEIGHT))
    # 3. Convertir de nuevo a 3 canales (BGR) porque el modelo YOLO lo espera así
    frame_para_yolo = cv2.cvtColor(frame_preprocesado, cv2.COLOR_GRAY2BGR)
    # --- FIN DEL PREPROCESAMIENTO ---

    # Realizar la inferencia en el frame preprocesado
    results = model(frame_para_yolo, conf=CONF, iou=IOU, imgsz=IMGSZ, verbose=False)

    # Copiamos el frame original para dibujar sobre él y no afectar la fuente original
    annotated_frame = frame_original.copy()

    # Calcular factores de escala para mapear las detecciones a la imagen original
    scale_w = original_w / PREPROCESS_WIDTH
    scale_h = original_h / PREPROCESS_HEIGHT

    # --- NUEVO: contadores y banderas para CSV ---
    num_personas = 0
    bandera_fumando = 0
    clases_fumar = {"smoke", "cigarette", "cigar", "fumando", "smoking"}  # ajustar a tus clases reales

    # Iterar sobre las detecciones encontradas
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id] if hasattr(model, "names") and cls_id in model.names else str(cls_id)

            # --- NUEVO: actualizar métricas ---
            if cls_name.lower() == "person":
                num_personas += 1
            if cls_name.lower() in clases_fumar:
                bandera_fumando = 1

            # Escalar las coordenadas al tamaño del frame original
            scaled_x1 = int(x1 * scale_w)
            scaled_y1 = int(y1 * scale_h)
            scaled_x2 = int(x2 * scale_w)
            scaled_y2 = int(y2 * scale_h)

            # Dibujar rectángulo
            cv2.rectangle(annotated_frame, (scaled_x1, scaled_y1), (scaled_x2, scaled_y2), (0, 255, 0), 2)
            label = f"{cls_name} {conf:.2f}"
            cv2.putText(annotated_frame, label, (scaled_x1, scaled_y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    # Calcular y mostrar FPS
    now = time.time()
    fps = 1.0 / max(now - prev_t, 1e-6)
    prev_t = now
    fps_smooth = fps if fps_smooth is None else (0.9 * fps_smooth + 0.1 * fps)
    cv2.putText(annotated_frame, f"FPS: {fps_smooth:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Mostrar índice de cámara actual
    cv2.putText(annotated_frame, f"Camara: {CAM_INDEX}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # --- NUEVO: guardar imagen y fila CSV por frame ---
    # Fecha/hora para CSV
    date_csv = datetime.now().strftime("%d/%m/%y")  # dd/mm/YY
    time_csv = datetime.now().strftime("%H:%M:%S")

    ruta_img = guardar_imagen_screenshot(annotated_frame)

    # Incrementar ID y guardar CSV
    ID_CONTADOR += 1
    guardar_fila_csv(
        ID_CONTADOR,
        date_csv,
        time_csv,
        ID_CAMARA_FIJO,
        ruta_img,
        num_personas,
        bandera_fumando
    )

    # Mostrar el resultado final
    cv2.imshow("YOLOv8 con Preprocesamiento", annotated_frame)

    # (Opcional) Escribir el frame al archivo de video
    if writer is not None:
        writer.write(annotated_frame)

    # Detectar teclas presionadas
    k = cv2.waitKey(1) & 0xFF

    # Presionar 'P' para cambiar de cámara
    if k == ord('p') or k == ord('P'):
        print(f"\n Cambiando de cámara...")

        # Liberar cámara actual
        cap.release()
        if writer is not None:
            writer.release()
            writer = None

        # Intentar abrir la siguiente cámara (ciclo de 0 a 5)
        intentos = 0
        max_camaras = 6
        while intentos < max_camaras:
            CAM_INDEX = (CAM_INDEX + 1) % max_camaras
            resultado = iniciar_camara(CAM_INDEX)

            if resultado is not None:
                cap, original_w, original_h = resultado

                # Reiniciar el escritor de video si estaba activado
                if SAVE_VIDEO:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, 20.0, (original_w, original_h))

                print(f" Cámara cambiada exitosamente al índice {CAM_INDEX}\n")
                break

            intentos += 1

        if intentos >= max_camaras:
            print(" No se encontró ninguna otra cámara disponible.")
            print(" Volviendo a la cámara inicial...")
            CAM_INDEX = 0
            resultado = iniciar_camara(CAM_INDEX)
            if resultado is None:
                print("Error crítico: No se pudo reabrir ninguna cámara.")
                break
            cap, original_w, original_h = resultado

    # Salir del bucle si se presiona 'q' o ESC
    if k == ord('q') or k == 27:
        print("Cerrando aplicación...")
        break

# --- LIMPIEZA FINAL ---
cap.release()
if writer is not None:
    writer.release()
cv2.destroyAllWindows()
