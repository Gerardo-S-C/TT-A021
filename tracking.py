import cv2
import time
import os
import torch
from ultralytics import YOLO

# --- CONFIGURACIÓN GENERAL ---

# >>> EDITA ESTAS RUTAS Y PARÁMETROS <<<
MODEL_PATH = 'runs/detect/yolo_v3/weights/best.pt'
CAM_INDEX  = 0

# --- PARÁMETROS DE INFERENCIA Y PROCESAMIENTO ---
# ## MODIFICACIÓN ##: IMGSZ es ahora la ÚNICA variable que controla la resolución.
# Define aquí el tamaño de entrada (alto, ancho) para el que fue entrenado tu modelo.
IMGSZ = (576, 736)  # Formato (alto, ancho)

CONF  = 0.60  # Confianza mínima
IOU   = 0.20  # IoU para NMS

# --- CONFIGURACIÓN DE VIDEO (OPCIONAL) ---
SAVE_VIDEO = False
OUTPUT_VIDEO_PATH = "salida_yolo.mp4"

# --------------------------------------------------------------------------

# 1. Cargar el modelo
assert os.path.exists(MODEL_PATH), f"Error: No se encontró el modelo en la ruta: {MODEL_PATH}"
model = YOLO(MODEL_PATH)

# 2. Seleccionar dispositivo de cómputo (GPU o CPU)
device = "cuda" if torch.cuda.is_available() and MODEL_PATH.endswith(".pt") else "cpu"
print(f"Usando modelo: '{MODEL_PATH}'")
print(f"Usando dispositivo: '{device}'")

# 3. Iniciar captura de video
cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
if not cap.isOpened():
    print(f"Error: No se pudo abrir la cámara con índice {CAM_INDEX}.")
    exit()

# (Opcional) Fijar una resolución de captura de la cámara
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print(f"Resolución de la cámara: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
print(f"Resolución para YOLO (ancho x alto): {IMGSZ[1]}x{IMGSZ[0]}")


# 4. (Opcional) Configurar el escritor de video si está activado
writer = None
if SAVE_VIDEO:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # ## MODIFICACIÓN ##: El tamaño del video se toma directamente de IMGSZ.
    # Nota: cv2.VideoWriter espera (ancho, alto), por eso se invierten los índices de IMGSZ.
    writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, 20.0, (IMGSZ[1], IMGSZ[0]))
    print(f"La salida de video se guardará en: '{OUTPUT_VIDEO_PATH}'")

# Variables para cálculo de FPS
prev_t = time.time()
fps_smooth = None

# --- BUCLE PRINCIPAL DE PROCESAMIENTO ---
while True:
    ok, frame_original = cap.read()
    if not ok:
        print("⚠️ No se pudo leer el fotograma de la cámara. Terminando...")
        break

    # --- PREPROCESAMIENTO ---
    frame_gris = cv2.cvtColor(frame_original, cv2.COLOR_BGR2GRAY)
    
    # ## MODIFICACIÓN ##: Se redimensiona usando directamente la variable IMGSZ.
    # Nota: cv2.resize espera (ancho, alto), por eso se invierten los índices de IMGSZ.
    frame_preprocesado = cv2.resize(frame_gris, (IMGSZ[1], IMGSZ[0]))
    
    frame_para_yolo = cv2.cvtColor(frame_preprocesado, cv2.COLOR_GRAY2BGR)

    # Realizar la inferencia
    results = model(frame_para_yolo, conf=CONF, iou=IOU, imgsz=IMGSZ, verbose=False)

    # Anotar el frame con las detecciones
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]

            cv2.rectangle(frame_para_yolo, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{cls_name} {conf:.2f}"
            cv2.putText(frame_para_yolo, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    # Calcular y mostrar FPS
    now = time.time()
    fps = 1.0 / max(now - prev_t, 1e-6)
    prev_t = now
    fps_smooth = fps if fps_smooth is None else (0.9 * fps_smooth + 0.1 * fps)
    cv2.putText(frame_para_yolo, f"FPS: {fps_smooth:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Mostrar el frame procesado
    cv2.imshow("Entrada del modelo YOLOv8", frame_para_yolo)

    if writer is not None:
        writer.write(frame_para_yolo)

    k = cv2.waitKey(1) & 0xFF
    if k == ord('q') or k == 27:
        print("Cerrando aplicación...")
        break

# --- LIMPIEZA FINAL ---
cap.release()
if writer is not None:
    writer.release()
cv2.destroyAllWindows()