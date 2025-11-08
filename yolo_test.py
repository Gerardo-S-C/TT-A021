from ultralytics import YOLO
import cv2
import os
import random

# --- CONFIGURACION ---
RUTA_MODELO = 'runs/detect/yolo_v3/weights/best.pt'
DIRECTORIO_FUENTE = 'imagenes/preprocesadas/2025-10-03'
UMBRAL_CONFIANZA = 0.7

try:
    lista_imagenes = [f for f in os.listdir(DIRECTORIO_FUENTE) if f.endswith(('.jpg', '.png'))]
    if not lista_imagenes:
        print(f"Error: No se encontraron imágenes en '{DIRECTORIO_FUENTE}'")
        exit()
    nombre_imagen_aleatoria = random.choice(lista_imagenes)
    RUTA_FUENTE = os.path.join(DIRECTORIO_FUENTE, nombre_imagen_aleatoria)
    print(f"Imagen seleccionada para evaluar: {RUTA_FUENTE}")
except FileNotFoundError:
    print(f"Error: El directorio '{DIRECTORIO_FUENTE}' no fue encontrado.")
    exit()

# --- PREDICCION Y VISUALIZACION ---

model = YOLO(RUTA_MODELO)
results = model.predict(source=RUTA_FUENTE, conf=UMBRAL_CONFIANZA, save=False, verbose=False)

# Obtener los nombres de las clases del modelo
nombres_clases = model.names

for r in results:
    img = cv2.imread(RUTA_FUENTE)
    overlay = img.copy()

    contador_personas = 0
    hay_cigarro = False

    boxes = r.boxes
    for box in boxes:
        # --- NUEVO: Dibujar la caja y la etiqueta de cada deteccion ---
        
        # Obtenemos las coordenadas de la caja
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # Convertir a enteros
        
        # Obtenemos la clase y la confianza
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        
        # Color y etiqueta
        nombre_clase = nombres_clases[cls]
        etiqueta = f'{nombre_clase} {conf:.2f}'
        
        # Asignar un color por clase
        color = (255, 0, 0) if nombre_clase == 'persona' else (0, 255, 255) # Azul para persona, Amarillo para cigarro

        # Dibujar el rectangulo en la imagen
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Dibujar el texto de la etiqueta justo encima de la caja
        cv2.putText(img, etiqueta, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # --- FIN DE LA SECCION NUEVA ---

        # Mantener la logica de conteo
        if cls == 0:
            contador_personas += 1
        elif cls == 1:
            hay_cigarro = True

    # Preparar el texto de resumen
    texto_personas = f"Personas: {contador_personas}"
    texto_cigarros = f"Cigarros: {'SI' if hay_cigarro else 'NO'}"
    
    # Dibujar el fondo semitransparente para el resumen
    cv2.rectangle(overlay, (10, 10), (250, 70), (0,0,0), -1)
    alpha = 0.6
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    # Dibujar el texto del resumen
    cv2.putText(img, texto_personas, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
    cv2.putText(img, texto_cigarros, (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
    
    # Guardar la imagen final
    nombre_salida = 'prediccion_con_cajas.jpg'
    cv2.imwrite(nombre_salida, img)
    print(f"Prediccion completada. Resultado guardado como '{nombre_salida}'")