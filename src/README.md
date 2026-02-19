# Módulos de Procesamiento de Firmas y Telegramas

Este directorio contiene todos los módulos necesarios para el procesamiento de telegramas electorales y verificación de firmas.

## 📁 Estructura de Archivos

### Dependencias y Utilidades Base
- **`all_deps.py`**: Todas las dependencias externas (numpy, cv2, PIL, PyTorch, sklearn, etc.)
- **`import_all.py`**: Hook de conveniencia para notebooks Jupyter que importa todo

### Procesamiento de Telegramas
- **`telegramas_crop_signatures.py`**: Pipeline completo para procesar telegramas electorales
  - Conversión de PDF a PNG
  - **Enderezamiento usando códigos QR** 
  - Recorte automático de regiones de firma
  - Clasificación de presencia de firmas
  
- **`telegram_tools.py`**: Herramientas auxiliares para procesamiento de imágenes
  - Procesamiento paralelo
  - Preprocesamiento de imágenes
  - Análisis de componentes conectados
  - Visualización

### Verificación de Firmas (Redes Siamesas)
- **`datasets.py`**: Cargadores de datasets (SIGCOMP2009, MNIST, CEDAR, Omniglot)
- **`models.py`**: Arquitecturas de redes neuronales (ResNet18 encoder, Siamese classifier)
- **`tools.py`**: Utilidades ML (binarización, detección de dispositivo, vocabulario)
- **`visualization.py`**: Funciones de visualización (pares, batches, métricas, PCA)
- **`pair_dataset.py`**: Dataset de pares para entrenamiento Siamese

## 🔄 Migración desde `deteccion-firmas-master/python-src`

Los siguientes archivos fueron migrados y actualizados:

| Archivo Original | Nuevo Nombre | Cambios |
|-----------------|--------------|---------|
| `import_all_universe.py` | ➡️ `all_deps.py` | Fusionado y expandido |
| `telegramas_crop_signatures.py` | ✅ `telegramas_crop_signatures.py` | Imports actualizados |
| `tools.py` | ➡️ `telegram_tools.py` | Renombrado para evitar conflicto |
| `import_all_notebook_hook.py` | ➡️ `import_all.py` | Renombrado y mantenido como hook principal |

## 🚀 Uso Rápido

### Para Notebooks Jupyter
```python
from src.import_all import *

# Ahora tienes acceso a todo:
# - Procesamiento de telegramas
# - Modelos de verificación de firmas
# - Herramientas de visualización
```

### Para Scripts Python

**Procesamiento de telegramas:**
```python
from src.telegramas_crop_signatures import straighten_image, crop_signatures
from src.telegram_tools import parallel_fmap
```

**Verificación de firmas:**
```python
from src.datasets import pds_sigcomp2009
from src.models import Encoder_Resnet18, SiamessClassifier
from src.pair_dataset import PairDataset
```

## 🔑 Características Clave

### Enderezamiento de Telegramas con QR
El módulo `telegramas_crop_signatures` incluye el código que **endereza telegramas electorales usando los códigos QR en las esquinas**:

- `extract_qr_coordinates()`: Detecta y extrae coordenadas de QR codes
- `straighten_image()`: Corrige rotación usando múltiples mediciones de ángulo
- `crop_by_qr()`: Recorta imagen usando límites de QR codes
- `ordered_points()`: Ordena puntos de QR para transformaciones geométricas

### Extracción Automática de Firmas
- Recorta 9 posiciones predefinidas (presidente, auxiliares, fiscales)
- Filtra texto impreso vs. manuscrito usando análisis de solidez
- Clasificador heurístico de presencia de firma

### Verificación de Firmas con Siamese Networks
- Encoder ResNet-18 preentrenado
- PairDataset para generación automática de pares matching/non-matching
- Transformaciones específicas para firmas (binarización, normalización)

## 📦 Dependencias Principales

Ver `all_deps.py` para lista completa. Principales:
- **OpenCV** (cv2): Procesamiento de imágenes
- **PyTorch**: Deep learning
- **pyzbar**: Detección de códigos QR
- **pdf2image**: Conversión de PDFs
- **scikit-image**: Análisis de componentes
- **scikit-learn**: PCA, métricas
- **imgaug**: Data augmentation

## 📝 Documentación

Todos los módulos incluyen docstrings completos siguiendo el formato Google/NumPy con:
- Descripción del propósito
- Argumentos con tipos
- Valores de retorno
- Ejemplos cuando es relevante

