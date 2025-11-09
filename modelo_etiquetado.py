import os
import cv2
import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom import minidom

# --- CONFIGURACION ---

# Ruta al MEJOR modelo entrenado (el archivo best.pt)
RUTA_MODELO = 'runs/detect/yolo_v3/weights/best.pt'

# Carpeta con el lote de imagenes
DIRECTORIO_LOTE = 'AZCONT_preprocesadas'

# Nombre del archivo XML de salida (en formato CVAT)
ARCHIVO_XML_SALIDA = 'annotations.xml'

# Parametros de prediccion
UMBRAL_CONFIANZA = 0.4  # Umbral para considerar una deteccion como válida
BATCH_SIZE = 8          # Numero de imagenes a procesar a la vez

def pre_etiquetar_y_convertir():
    """
    Usa el modelo YOLO para pre-etiquetar un lote de imagenes y generar
    directamente un archivo XML en formato CVAT.
    Esta version procesa en "super-lotes" para evitar errores de memoria.
    """
    # Cargar modelo y preparar imagenes
    print(f"--- FASE 1: Cargando modelo y preparando lista de imagenes ---")
    try:
        model = YOLO(RUTA_MODELO)
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return

    try:
        rutas_imagenes = [os.path.join(DIRECTORIO_LOTE, f) for f in os.listdir(DIRECTORIO_LOTE) if f.endswith(('.jpg', '.png'))]
        if not rutas_imagenes:
            print(f"No se encontraron imagenes en '{DIRECTORIO_LOTE}'.")
            return
    except FileNotFoundError:
        print(f"Error: No se encontro el directorio '{DIRECTORIO_LOTE}'.")
        return

    # Procesar en "super-lotes"
    SUPER_BATCH_SIZE = 150 # Tamaño de los grupos de imagenes a procesar
    todas_las_detecciones = []
    
    print(f"Se procesaran {len(rutas_imagenes)} imagenes en grupos de {SUPER_BATCH_SIZE}.")

    for i in tqdm(range(0, len(rutas_imagenes), SUPER_BATCH_SIZE), desc="Procesando Super-Lotes"):
        # Obtener el grupo actual de imagenes
        lote_actual = rutas_imagenes[i:i + SUPER_BATCH_SIZE]
        
        # Realizar prediccion SOLO en el grupo actual
        results = model.predict(source=lote_actual, conf=UMBRAL_CONFIANZA, batch=BATCH_SIZE, verbose=False)
        
        # Procesar los resultados del grupo
        for result in results:
            nombre_archivo = os.path.basename(result.path)
            for box in result.boxes:
                x_c, y_c, w, h = box.xywhn[0]
                todas_las_detecciones.append({
                    'filename': nombre_archivo, 'class': model.names[int(box.cls[0])],
                    'x_center': float(x_c), 'y_center': float(y_c),
                    'width': float(w), 'height': float(h)
                })
    
    if not todas_las_detecciones:
        print("El modelo no detecto ningún objeto.")
        return
        
    df_predicciones = pd.DataFrame(todas_las_detecciones)
    print(f"\nPrediccion completada. Se encontraron {len(df_predicciones)} objetos.")

    # Convertir predicciones a XML de CVAT
    print(f"\n--- FASE 2: Convirtiendo predicciones a formato XML ---")
    grouped = df_predicciones.groupby('filename')
    root = Element('annotations')
    SubElement(root, 'version').text = '1.1'

    image_id = 0
    for filename, group in tqdm(grouped, desc="Generando XML"):
        img_path = os.path.join(DIRECTORIO_LOTE, filename)
        img = cv2.imread(img_path)
        if img is None: continue
        height, width, _ = img.shape

        image_tag = SubElement(root, 'image', id=str(image_id), name=filename, width=str(width), height=str(height))
        
        for _, row in group.iterrows():
            xtl = (row['x_center'] - row['width'] / 2) * width
            ytl = (row['y_center'] - row['height'] / 2) * height
            xbr = (row['x_center'] + row['width'] / 2) * width
            ybr = (row['y_center'] + row['height'] / 2) * height
            
            SubElement(image_tag, 'box', label=row['class'], occluded='0', xtl=f"{xtl:.2f}", ytl=f"{ytl:.2f}", xbr=f"{xbr:.2f}", ybr=f"{ybr:.2f}")
        
        image_id += 1

    xml_str = tostring(root)
    pretty_xml_str = minidom.parseString(xml_str).toprettyxml(indent="  ")
    
    ruta_xml_final = os.path.join(DIRECTORIO_LOTE, ARCHIVO_XML_SALIDA)
    with open(ruta_xml_final, "w") as f:
        f.write(pretty_xml_str)
        
    print(f"\n¡Proceso completo!")
    print(f"El archivo de anotaciones se ha guardado en: '{ruta_xml_final}'")


if __name__ == '__main__':
    pre_etiquetar_y_convertir()