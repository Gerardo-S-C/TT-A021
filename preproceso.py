import os
import cv2
from tqdm import tqdm

# --- CONFIGURACION DE RUTAS Y PARAMETROS ---

# Directorio de las imagenes originales
#DIRECTORIO_ENTRADA = 'imagenes/2025-10-03'
DIRECTORIO_ENTRADA = 'azureConteiner'

# Directorio donde se guardaran las imagenes procesadas
#DIRECTORIO_SALIDA = 'imagenes/preprocesadas/2025-10-03'
DIRECTORIO_SALIDA = 'AZCONT_preprocesadas'

# 3. Dimensiones para las imagenes
ANCHO_SALIDA = 736
ALTO_SALIDA = 576

def preprocesar_imagenes(input_dir, output_dir, width, height):
    """
    Lee imagenes de un directorio, las convierte a escala de grises,
    las redimensiona y las guarda en un nuevo directorio.
    """
    # Crea el directorio de salida si no existe
    # exist_ok=True evita un error si la carpeta ya ha sido creada
    os.makedirs(output_dir, exist_ok=True)
    print(f"Directorio de salida '{output_dir}' asegurado.")

    # Obtiene la lista de imagenes a procesar
    try:
        extensiones_validas = ['.jpg', '.png']
        nombres_archivos = [
            f for f in os.listdir(input_dir) 
            if os.path.splitext(f)[1].lower() in extensiones_validas
        ]
        if not nombres_archivos:
            print(f"Error: No se encontraron imagenes en el directorio '{input_dir}'.")
            return
    except FileNotFoundError:
        print(f"Error: El directorio de entrada '{input_dir}' no existe.")
        return

    print(f"Se encontraron {len(nombres_archivos)} imagenes para procesar.")

    # Itera sobre cada imagen, la procesa y la guarda
    # tqdm para tener una barra de progreso
    for nombre_archivo in tqdm(nombres_archivos, desc="Procesando imagenes"):
        try:
            # Construir las rutas de entrada y salida
            ruta_entrada = os.path.join(input_dir, nombre_archivo)
            ruta_salida = os.path.join(output_dir, nombre_archivo)

            # Leer la imagen
            imagen = cv2.imread(ruta_entrada)

            if imagen is not None:
                # Convertir a escala de grises
                imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
                
                # Redimensionar a la resolucion deseada
                imagen_redimensionada = cv2.resize(imagen_gris, (width, height))
                
                # Guardar la imagen procesada
                cv2.imwrite(ruta_salida, imagen_redimensionada)
            else:
                print(f"\nAdvertencia: No se pudo leer el archivo '{nombre_archivo}', se omitira.")

        except Exception as e:
            print(f"\nError al procesar el archivo '{nombre_archivo}': {e}")
            continue
    
    print("\n¡Preprocesamiento completado con exito!")
    print(f"Todas las imagenes han sido guardadas en '{output_dir}'.")


if __name__ == "__main__":
    preprocesar_imagenes(
        input_dir=DIRECTORIO_ENTRADA,
        output_dir=DIRECTORIO_SALIDA,
        width=ANCHO_SALIDA,
        height=ALTO_SALIDA
    )