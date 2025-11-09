# TT-A021 - Sistema de Detección de Personas y Objetos de Fumar con YOLO

## Descripción

Este proyecto es un sistema de detección de objetos basado en YOLO (You Only Look Once) diseñado para detectar personas y objetos relacionados con fumar (cigarrillos, humo) en imágenes térmicas y video en tiempo real. El sistema utiliza modelos YOLOv8 entrenados personalmente y proporciona herramientas completas para el procesamiento, entrenamiento y análisis de datos.

## Características Principales

- **Detección en tiempo real**: Captura video desde cámara y detecta objetos en tiempo real
- **Procesamiento por lotes**: Procesa múltiples imágenes de forma eficiente
- **Entrenamiento de modelos**: Herramientas para entrenar modelos YOLO personalizados
- **Preprocesamiento de imágenes**: Conversión a escala de grises y redimensionado automático
- **Almacenamiento de datos**: Guarda resultados en CSV con metadatos de detección
- **Pre-etiquetado automático**: Genera anotaciones XML en formato CVAT usando modelos entrenados
- **Organización de datasets**: Divide automáticamente datasets en conjuntos de entrenamiento y validación

## Requisitos

### Dependencias de Python

- `ultralytics` - Framework YOLO
- `opencv-python` (cv2) - Procesamiento de imágenes y video
- `torch` (PyTorch) - Framework de deep learning
- `pandas` - Manipulación de datos
- `tqdm` - Barras de progreso
- `numpy` - Operaciones numéricas (generalmente incluido con las anteriores)

### Hardware Recomendado

- GPU NVIDIA con CUDA (opcional pero recomendado para entrenamiento)
- Cámara USB o térmica para detección en tiempo real
- Mínimo 8GB de RAM

## Instalación

1. **Clonar el repositorio**:
```bash
git clone <url-del-repositorio>
cd TT-A021
```

2. **Instalar dependencias**:
```bash
pip install ultralytics opencv-python torch pandas tqdm numpy
```

O crear un archivo `requirements.txt` con:
```
ultralytics
opencv-python
torch
pandas
tqdm
numpy
```

Y luego ejecutar:
```bash
pip install -r requirements.txt
```

3. **Verificar instalación de CUDA** (opcional para GPU):
```python
import torch
print(f"CUDA disponible: {torch.cuda.is_available()}")
```

## Estructura del Proyecto

```
TT-A021/
├── tracking.py              # Detección en tiempo real desde cámara
├── video.py                 # Procesamiento por lotes de imágenes
├── yolo_train_test.py       # Script de entrenamiento del modelo
├── yolo_test.py             # Prueba del modelo en imagen aleatoria
├── preproceso.py            # Preprocesamiento de imágenes
├── organizar.py             # Organización de dataset (train/val)
├── selec.py                 # Selección aleatoria de imágenes
├── modelo_etiquetado.py     # Pre-etiquetado automático
└── README.md                # Este archivo
```

## Descripción de Archivos

### `tracking.py`
Script principal para detección en tiempo real. Captura video desde una cámara, preprocesa los frames (escala de grises y redimensionado), ejecuta inferencia con YOLO y guarda:
- Imágenes anotadas con detecciones
- Archivo CSV con metadatos (ID, fecha, hora, número de personas, bandera de fumar)
- Video de salida (opcional)

**Características**:
- Cambio de cámara con tecla 'P'
- Visualización de FPS en tiempo real
- Guardado automático de screenshots
- Escalado de coordenadas para mapear detecciones al frame original

### `video.py`
Procesa un lote de imágenes preprocesadas, aplica detección YOLO y guarda las imágenes anotadas con bounding boxes y etiquetas.

**Uso**: Ideal para procesar secuencias de video frame por frame o grandes colecciones de imágenes.

### `yolo_train_test.py`
Script para entrenar un modelo YOLO personalizado. Carga un modelo pre-entrenado y lo fine-tunea con datos personalizados.

**Configuración requerida**:
- Archivo YAML con configuración del dataset (`dataset_yolov4/data.yaml`)
- Modelo base YOLO (`.pt`)

### `yolo_test.py`
Prueba rápida del modelo seleccionando una imagen aleatoria de un directorio y mostrando las detecciones con bounding boxes.

### `preproceso.py`
Preprocesa imágenes convirtiéndolas a escala de grises y redimensionándolas a dimensiones específicas (736x576 por defecto). Útil para preparar datos antes del entrenamiento o inferencia.

### `organizar.py`
Organiza un dataset dividiéndolo en conjuntos de entrenamiento y validación según el formato requerido por YOLO. Crea la estructura de carpetas `train/images`, `train/labels`, `val/images`, `val/labels`.

### `selec.py`
Selecciona un número específico de imágenes aleatorias de un directorio y las copia a otro directorio. Útil para crear muestras representativas del dataset.

### `modelo_etiquetado.py`
Utiliza un modelo YOLO entrenado para pre-etiquetar automáticamente un lote de imágenes y generar un archivo XML en formato CVAT. Procesa imágenes en super-lotes para evitar problemas de memoria.

## Uso

### Detección en Tiempo Real

```bash
python tracking.py
```

**Antes de ejecutar**, edita las siguientes variables en `tracking.py`:
- `MODEL_PATH`: Ruta al modelo entrenado (`.pt`)
- `CAM_INDEX`: Índice de la cámara (0 = cámara por defecto)
- `PREPROCESS_WIDTH` y `PREPROCESS_HEIGHT`: Dimensiones de preprocesamiento
- `CONF`: Umbral de confianza (0.0-1.0)
- `IOU`: Umbral IoU para NMS
- `ID_CAMARA_FIJO`: Identificador de la cámara

**Controles**:
- `P`: Cambiar de cámara
- `Q` o `ESC`: Salir

**Salidas**:
- Carpeta `imagenes/`: Screenshots anotados
- Carpeta `datos_csv/`: Archivo `resultados.csv` con metadatos
- Video `vela1persona.mp4` (si `SAVE_VIDEO = True`)

### Procesamiento por Lotes

```bash
python video.py
```

**Configuración en `video.py`**:
- `RUTA_MODELO`: Ruta al modelo entrenado
- `DIRECTORIO_FUENTE`: Carpeta con imágenes a procesar
- `DIRECTORIO_SALIDA`: Carpeta donde guardar resultados
- `UMBRAL_CONFIANZA`: Umbral de confianza mínimo

### Entrenamiento del Modelo

```bash
python yolo_train_test.py
```

**Requisitos previos**:
1. Dataset organizado en formato YOLO (usar `organizar.py`)
2. Archivo `data.yaml` con configuración del dataset
3. Modelo base YOLO (puede ser un modelo pre-entrenado o uno anterior)

**Configuración en `yolo_train_test.py`**:
- `ARCHIVO_YAML`: Ruta al archivo de configuración
- `RUTA_MODELO`: Modelo base para fine-tuning
- Parámetros de entrenamiento: `epochs`, `batch`, `imgsz`, `patience`

### Preprocesamiento de Imágenes

```bash
python preproceso.py
```

**Configuración**:
- `DIRECTORIO_ENTRADA`: Carpeta con imágenes originales
- `DIRECTORIO_SALIDA`: Carpeta para imágenes preprocesadas
- `ANCHO_SALIDA` y `ALTO_SALIDA`: Dimensiones objetivo

### Organización de Dataset

```bash
python organizar.py
```

**Configuración**:
- `DIRECTORIO_FUENTE_IMAGENES`: Carpeta con imágenes originales
- `DIRECTORIO_FUENTE_LABELS`: Carpeta con etiquetas (archivos `.txt`)
- `DIRECTORIO_SALIDA`: Carpeta base para la nueva estructura
- `PORCENTAJE_TRAIN`: Porcentaje para entrenamiento (0.8 = 80%)

### Selección Aleatoria de Imágenes

```bash
python selec.py
```

**Configuración**:
- `DIRECTORIO_ENTRADA`: Carpeta fuente
- `DIRECTORIO_SALIDA`: Carpeta destino
- `NUM_IMAGENES_A_SELECCIONAR`: Número de imágenes a seleccionar

### Pre-etiquetado Automático

```bash
python modelo_etiquetado.py
```

**Configuración**:
- `RUTA_MODELO`: Modelo entrenado para usar
- `DIRECTORIO_LOTE`: Carpeta con imágenes a etiquetar
- `UMBRAL_CONFIANZA`: Umbral mínimo de confianza
- `BATCH_SIZE`: Tamaño de lote para procesamiento

**Salida**: Archivo `annotations.xml` en formato CVAT

## Configuración Importante

### Rutas de Modelos

Los scripts esperan modelos en formato `.pt` (PyTorch). Asegúrate de tener el modelo entrenado antes de ejecutar scripts de inferencia.

**Rutas comunes**:
- `runs/detect/yolo_v3/weights/best.pt`
- `runs/detect/yolo_v4/weights/best.pt`

### Formato de Datos

- **Imágenes**: `.jpg` o `.png`
- **Etiquetas YOLO**: Archivos `.txt` con formato `class_id x_center y_center width height` (normalizado)
- **Configuración YAML**: Formato estándar de YOLO con rutas a train/val

### Dimensiones de Imagen

El proyecto está configurado para trabajar con imágenes de **736x576** píxeles. Asegúrate de que tus modelos estén entrenados con estas dimensiones o ajusta los parámetros según corresponda.

### Clases Detectadas

El modelo detecta las siguientes clases (verificar según tu modelo entrenado):
- `person` / `persona`: Personas
- `smoke` / `cigarette` / `cigar` / `fumando` / `smoking`: Objetos relacionados con fumar

## Notas Adicionales

### Rendimiento

- **GPU**: El uso de GPU acelera significativamente el entrenamiento y la inferencia
- **Preprocesamiento**: El preprocesamiento a escala de grises puede mejorar el rendimiento en imágenes térmicas
- **Batch Size**: Ajusta según la memoria disponible de tu GPU

### Almacenamiento

- Las imágenes procesadas se guardan automáticamente con nombres basados en fecha
- El CSV de resultados se actualiza incrementalmente (no sobrescribe datos anteriores)
- Los videos de salida usan códec `mp4v`

### Troubleshooting

**Error al abrir cámara**:
- Verifica que la cámara esté conectada
- Prueba diferentes índices de cámara (0, 1, 2, etc.)
- En Windows, asegúrate de usar `cv2.CAP_DSHOW`

**Error de memoria**:
- Reduce el `BATCH_SIZE` en scripts de procesamiento por lotes
- Reduce las dimensiones de imagen si es necesario
- Cierra otras aplicaciones que usen GPU

**Modelo no encontrado**:
- Verifica la ruta al archivo `.pt`
- Asegúrate de que el archivo existe y es accesible

## Autores


- [@Remotepine99](https://github.com/REMOTEpine12) - Israel Díaz
- [@AtzMax](https://github.com/AtzMax) - Atzin Ignacio
- [@Gerardo-S-C](https://github.com/Gerardo-S-C) - Gerardo Sandoval


