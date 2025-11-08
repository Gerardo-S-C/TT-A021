import os
import random
import shutil
from tqdm import tqdm

# --- CONFIGURACION ---

# Directorio de TODAS las imagenes preprocesadas
DIRECTORIO_ENTRADA = 'imagenes/preprocesadas/2025-09-23'

# Directorio donde se guardara la muestra aleatoria
DIRECTORIO_SALIDA = 'lote3k'

# Imagenes aleatorias a seleccionar
NUM_IMAGENES_A_SELECCIONAR = 3000

def crear_muestra_aleatoria(input_dir, output_dir, num_muestras):
    """
    Selecciona un numero de imagenes al azar de un directorio
    y las copia a un nuevo directorio.
    """
    # Crear el directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)
    print(f"Directorio de salida '{output_dir}' listo.")

    try:
        extensiones_validas = ['.jpg', '.png']
        imagenes_disponibles = [
            f for f in os.listdir(input_dir)
            if os.path.splitext(f)[1].lower() in extensiones_validas
        ]
    except FileNotFoundError:
        print(f"Error: No se encontro el directorio de entrada '{input_dir}'.")
        return

    if not imagenes_disponibles:
        print(f"Error: No hay imagenes en '{input_dir}'.")
        return

    # No pedir mas imagenes de las que hay disponibles
    if num_muestras > len(imagenes_disponibles):
        print(f"Advertencia: Se solicitaron {num_muestras} imagenes, pero solo hay {len(imagenes_disponibles)}.")
        num_muestras = len(imagenes_disponibles)

    # Seleccionar las imagenes al azar
    seleccion = random.sample(imagenes_disponibles, num_muestras)
    print(f"Se han seleccionado {len(seleccion)} imagenes al azar.")

    # Copiar las imagenes seleccionadas al directorio de salida
    for nombre_archivo in tqdm(seleccion, desc="Copiando imagenes"):
        ruta_origen = os.path.join(input_dir, nombre_archivo)
        ruta_destino = os.path.join(output_dir, nombre_archivo)
        shutil.copy(ruta_origen, ruta_destino)
        
    print("\n¡Muestra creada con exito!")
    print(f"Puedes encontrar tus {num_muestras} imagenes de prueba en la carpeta '{output_dir}'.")


# --- EJECUCIÓN DEL SCRIPT ---
if __name__ == "__main__":
    crear_muestra_aleatoria(
        input_dir=DIRECTORIO_ENTRADA,
        output_dir=DIRECTORIO_SALIDA,
        num_muestras=NUM_IMAGENES_A_SELECCIONAR
    )