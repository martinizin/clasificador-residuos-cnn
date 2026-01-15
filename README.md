# ♻️ Clasificador de Residuos con CNN y Streamlit

> **Proyecto de Inteligencia Artificial I** — Clasificación automática de imágenes de residuos en 6 categorías utilizando Transfer Learning con MobileNetV2 y despliegue interactivo con Streamlit.

---

## 📋 Tabla de Contenidos

1. [Resumen Ejecutivo](#resumen-ejecutivo)
2. [Arquitectura de la Solución](#arquitectura-de-la-solución)
3. [Requisitos e Instalación](#requisitos-e-instalación)
4. [Preparación del Dataset](#preparación-del-dataset)
5. [Entrenamiento del Modelo](#entrenamiento-del-modelo)
6. [Fundamentos Teóricos](#fundamentos-teóricos)
7. [Explicación Detallada del Código](#explicación-detallada-del-código)
8. [Ejecución Local](#ejecución-local)
9. [Despliegue en Streamlit Cloud](#despliegue-en-streamlit-cloud)

---

## Resumen Ejecutivo

### ¿Qué hace este proyecto?

Este proyecto implementa un **sistema de clasificación automática de residuos** basado en imágenes. Dado una fotografía de un objeto (cartón, vidrio, metal, papel, plástico o basura genérica), el modelo predice a cuál de las **6 categorías** pertenece.

### ¿A quién sirve?

- **Estudiantes** aprendiendo sobre redes neuronales convolucionales (CNN) y Transfer Learning.
- **Desarrolladores** que desean un ejemplo práctico de ML end-to-end (entrenamiento + despliegue).
- **Proyectos ambientales** que necesiten automatizar la clasificación de residuos.

### Clases que clasifica el modelo

| Clase       | Descripción                        |
|-------------|------------------------------------|
| `cardboard` | Cartón (cajas, empaques)           |
| `glass`     | Vidrio (botellas, frascos)         |
| `metal`     | Metal (latas, aluminio)            |
| `paper`     | Papel (hojas, periódicos)          |
| `plastic`   | Plástico (botellas, envases)       |
| `trash`     | Basura genérica (no reciclable)    |

### Demo

Al ejecutar la aplicación, verás una interfaz como esta:

```
♻️ Clasificador de Residuos (6 clases)
Sube una imagen y el modelo predice: cardboard, glass, metal, paper, plastic, trash.

[📁 Sube una imagen (JPG/PNG)]

Predicción final: plastic — probabilidad: 0.9234
Top 3:
- plastic: 0.9234
- glass: 0.0521
- metal: 0.0189

[Gráfico de barras con todas las probabilidades]
```

---

## Arquitectura de la Solución

El proyecto sigue un **pipeline clásico de Machine Learning**:

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              PIPELINE COMPLETO                                       │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│   📁 Dataset        🔄 Preprocesamiento      🧠 Entrenamiento      📊 Evaluación    │
│   (Kaggle)    ───►  (Resize, Split,    ───►  (MobileNetV2 +   ───► (Accuracy,       │
│   6 carpetas        Augmentation)            Transfer Learning)    Val Loss)        │
│                                                                                      │
│        │                                                                             │
│        ▼                                                                             │
│                                                                                      │
│   💾 Exportación       🔮 Inferencia         🌐 Streamlit         ☁️ Deploy         │
│   (model.keras,   ◄─── (Predicción      ◄─── (Interfaz web)  ◄─── (Streamlit        │
│    labels.json)        con imagen)                                  Cloud)          │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### Diagrama de archivos generados

```
Entrenamiento (train.py)              Inferencia (app.py)
         │                                   │
         ▼                                   ▼
   ┌─────────────┐                    ┌─────────────┐
   │ model.keras │ ◄───────────────── │  Cargar     │
   │ labels.json │                    │  modelo     │
   │ meta.json   │                    └─────────────┘
   └─────────────┘                           │
                                             ▼
                                    ┌─────────────────┐
                                    │ Predicción      │
                                    │ + Top-3         │
                                    │ + Gráfico       │
                                    └─────────────────┘
```

---

## Requisitos e Instalación

### Prerrequisitos

| Requisito      | Versión mínima | Verificar con           |
|----------------|----------------|-------------------------|
| Python         | 3.9+           | `python --version`      |
| pip            | 21.0+          | `pip --version`         |
| Git (opcional) | 2.0+           | `git --version`         |

### Estructura de carpetas del proyecto

```
modeloIA/
├── data/
│   └── raw/
│       ├── cardboard/    # ~400 imágenes
│       ├── glass/        # ~500 imágenes
│       ├── metal/        # ~400 imágenes
│       ├── paper/        # ~590 imágenes
│       ├── plastic/      # ~480 imágenes
│       └── trash/        # ~130 imágenes
├── model/
│   ├── model.keras       # Modelo entrenado
│   ├── labels.json       # ["cardboard", "glass", ...]
│   └── meta.json         # {"img_size": 224, "arch": "MobileNetV2"}
├── train.py              # Script de entrenamiento
├── app.py                # Aplicación Streamlit
├── requirements.txt      # Dependencias
└── README.md             # Este archivo
```

### Paso 1: Clonar o descargar el repositorio

```bash
# Si tienes Git:
git clone https://github.com/tu-usuario/recycle-cnn.git
cd recycle-cnn

# Si descargaste ZIP: descomprimir y abrir carpeta en terminal
```

### Paso 2: Crear entorno virtual

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

> ⚠️ **Si da error de políticas de ejecución**, ejecuta primero:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

**Windows (CMD):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Paso 3: Instalar dependencias

```bash
pip install -r requirements.txt
```

Contenido de `requirements.txt`:
```
streamlit
tensorflow
pillow
numpy
```

### Paso 4: Verificar instalación

```bash
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"
python -c "import streamlit; print(f'Streamlit: {streamlit.__version__}')"
```

---

## Preparación del Dataset

### Fuente del dataset

El dataset proviene de Kaggle: **[Garbage Classification (6 classes)](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification)**.

Contiene aproximadamente **2,500 imágenes** distribuidas en 6 clases.

### Descarga manual

1. Ir a Kaggle y descargar el dataset (requiere cuenta gratuita).
2. Descomprimir el archivo ZIP.
3. Organizar las imágenes en la estructura `data/raw/`:

```
data/
└── raw/
    ├── cardboard/
    │   ├── cardboard1.jpg
    │   ├── cardboard2.jpg
    │   └── ...
    ├── glass/
    │   ├── glass1.jpg
    │   └── ...
    ├── metal/
    ├── paper/
    ├── plastic/
    └── trash/
```

### Validar estructura

Ejecuta este comando para verificar que las carpetas existen:

**Windows:**
```powershell
Get-ChildItem data\raw -Directory | ForEach-Object { 
    Write-Host "$($_.Name): $((Get-ChildItem $_.FullName -File).Count) imágenes" 
}
```

**Linux/Mac:**
```bash
for dir in data/raw/*/; do echo "$(basename $dir): $(ls -1 $dir | wc -l) imágenes"; done
```

Salida esperada (aproximada):
```
cardboard: 403 imágenes
glass: 501 imágenes
metal: 410 imágenes
paper: 594 imágenes
plastic: 482 imágenes
trash: 137 imágenes
```

---

## Entrenamiento del Modelo

### Comando básico

```bash
python train.py
```

Esto usa los valores por defecto:
- `--data_dir`: `data/raw`
- `--img_size`: `224`
- `--batch`: `32`
- `--epochs`: `5`
- `--fine_tune_epochs`: `3`

### Comando con parámetros personalizados

```bash
python train.py --data_dir data/raw --img_size 224 --batch 32 --epochs 10 --fine_tune_epochs 5
```

### Hiperparámetros explicados

| Parámetro          | Valor default | Descripción                                                                 |
|--------------------|---------------|-----------------------------------------------------------------------------|
| `--data_dir`       | `data/raw`    | Carpeta que contiene las subcarpetas de clases                              |
| `--img_size`       | `224`         | Tamaño al que se redimensionan las imágenes (224×224 píxeles)               |
| `--batch`          | `32`          | Número de imágenes procesadas en paralelo por iteración                     |
| `--epochs`         | `5`           | Épocas de entrenamiento con la base congelada (solo cabeza)                 |
| `--fine_tune_epochs`| `3`          | Épocas adicionales con la base descongelada (fine-tuning)                   |

### Archivos generados

Después del entrenamiento, la carpeta `model/` contendrá:

| Archivo         | Contenido                                                    |
|-----------------|--------------------------------------------------------------|
| `model.keras`   | Modelo completo (arquitectura + pesos) en formato Keras 3    |
| `labels.json`   | Lista ordenada de clases: `["cardboard", "glass", ...]`      |
| `meta.json`     | Metadatos: `{"img_size": 224, "arch": "MobileNetV2"}`        |

**¿Por qué guardar `labels.json` y `meta.json`?**

Para asegurar **consistencia** entre entrenamiento e inferencia:
- El orden de las clases puede variar si se re-entrena en otra máquina.
- El tamaño de imagen debe coincidir exactamente.

---

## Fundamentos Teóricos

Esta sección explica los conceptos clave para **defender el proyecto** ante un jurado.

### ¿Qué es una CNN (Red Neuronal Convolucional)?

Una **CNN** es un tipo de red neuronal diseñada específicamente para procesar datos con estructura de grilla, como imágenes.

**¿Por qué CNN para imágenes?**

1. **Extracción jerárquica de características**: Las primeras capas detectan bordes y texturas; las profundas detectan formas y objetos.
2. **Invariancia espacial**: Puede detectar un objeto sin importar dónde esté en la imagen.
3. **Reducción de parámetros**: Usa convoluciones en lugar de conexiones densas, reduciendo la memoria necesaria.

```
Imagen (224×224×3) → Convoluciones → Pooling → ... → Features → Dense → Softmax → Clase
```

### ¿Qué es Transfer Learning?

**Transfer Learning** es la técnica de reutilizar un modelo entrenado en un problema similar para resolver uno nuevo.

**Analogía**: Es como si un chef experto en cocina italiana aprendiera cocina japonesa. No parte de cero; ya sabe técnicas de corte, tiempos de cocción, etc.

**¿Por qué usarlo?**
- Nuestro dataset es pequeño (~2,500 imágenes).
- Entrenar una CNN desde cero requeriría millones de imágenes.
- MobileNetV2 ya "sabe" detectar texturas, formas y objetos genéricos.

### ¿Por qué MobileNetV2?

| Característica        | MobileNetV2              | VGG16           | ResNet50       |
|-----------------------|--------------------------|-----------------|----------------|
| Parámetros            | ~3.4M                    | ~138M           | ~25.6M         |
| Tamaño del archivo    | ~14 MB                   | ~528 MB         | ~98 MB         |
| Velocidad en CPU      | Rápida                   | Lenta           | Media          |
| Precisión en ImageNet | 71.8%                    | 71.3%           | 74.9%          |

**Ventajas de MobileNetV2:**
- Ligero: ideal para despliegue en la nube (límites de Streamlit Cloud).
- Rápido: buena experiencia de usuario.
- Eficiente: usa "depthwise separable convolutions" que reducen cómputo.

### ¿Qué es ImageNet?

**ImageNet** es un dataset de ~14 millones de imágenes etiquetadas en ~22,000 categorías. El subconjunto ILSVRC tiene 1,000 clases (animales, objetos, vehículos, etc.).

Al usar `weights="imagenet"`, cargamos pesos de MobileNetV2 entrenada en este dataset. Estos pesos codifican conocimiento visual general que transferimos a nuestro problema.

### ¿Qué es Softmax?

**Softmax** es una función de activación que convierte un vector de valores reales en una distribución de probabilidad.

```
Entradas (logits):  [2.0, 1.0, 0.1, 0.5, 3.0, 0.8]
Salidas (probs):    [0.10, 0.04, 0.01, 0.02, 0.79, 0.03]
                                 │
                                 └── Suman 1.0
```

Se usa en clasificación multiclase porque:
- Las probabilidades son interpretables (79% confianza en "plastic").
- Permite calcular el top-3 de predicciones.

### ¿Qué es Sparse Categorical Crossentropy?

Es la **función de pérdida** que mide qué tan "equivocado" está el modelo.

**¿Por qué "sparse"?**
- Nuestras etiquetas son enteros: `0, 1, 2, 3, 4, 5`.
- Si fueran one-hot (`[0,0,1,0,0,0]`), usaríamos `categorical_crossentropy`.
- "Sparse" evita convertir a one-hot, ahorrando memoria.

**Fórmula simplificada:**
```
Loss = -log(probabilidad de la clase correcta)
```

Si el modelo predice 0.9 para la clase correcta: `loss = -log(0.9) ≈ 0.10` (bajo).
Si predice 0.1: `loss = -log(0.1) ≈ 2.30` (alto).

### ¿Qué es Overfitting y cómo lo evitamos?

**Overfitting** ocurre cuando el modelo memoriza los datos de entrenamiento pero no generaliza a datos nuevos.

**Señales de overfitting:**
- Accuracy de entrenamiento: 99%
- Accuracy de validación: 60%

**Técnicas de mitigación usadas en este proyecto:**

| Técnica          | Dónde se aplica           | Efecto                                        |
|------------------|---------------------------|-----------------------------------------------|
| Data Augmentation| `RandomFlip`, `RandomRotation`, `RandomZoom` | Genera variaciones artificiales de imágenes |
| Dropout          | `Dropout(0.2)`            | Apaga 20% de neuronas aleatoriamente          |
| Transfer Learning| Base congelada inicialmente| Aprovecha features pre-aprendidos            |
| Fine-tuning      | Learning rate muy bajo (1e-5)| Ajusta pesos sin destruir conocimiento      |

### Epochs, Batch Size y Learning Rate

| Concepto      | Definición                                      | Valor usado    |
|---------------|-------------------------------------------------|----------------|
| **Epoch**     | Una pasada completa por todo el dataset          | 5 + 3          |
| **Batch Size**| Número de imágenes procesadas antes de actualizar pesos | 32       |
| **Learning Rate** | Qué tan grande es el "paso" al ajustar pesos | 1e-3 → 1e-5   |

**Analogía del learning rate:**
- Muy alto (0.1): Caminas dando saltos enormes; puedes pasar de largo el mínimo.
- Muy bajo (1e-7): Caminas milímetro a milímetro; nunca llegas.
- Justo (1e-3 a 1e-5): Pasos razonables hacia el objetivo.

### Métricas: Accuracy

**Accuracy** = (predicciones correctas) / (total de predicciones) × 100

Ejemplo: Si de 100 imágenes, 85 se clasifican bien → Accuracy = 85%.

**Limitación**: Si el dataset está desbalanceado (trash tiene pocas imágenes), accuracy puede engañar.

**Mejoras sugeridas:**
- **Confusion Matrix**: Muestra errores por clase.
- **Precision/Recall/F1**: Métricas más robustas para clases desbalanceadas.

---

## Explicación Detallada del Código

### 📄 train.py — Bloque por Bloque

#### Bloque 1: Imports y configuración

```python
# train.py
import argparse          # Para leer argumentos de línea de comandos
import json              # Para guardar labels.json y meta.json
import os                # Para crear carpetas (os.makedirs)
import tensorflow as tf  # Framework de deep learning
from tensorflow import keras  # API de alto nivel para redes neuronales
```

#### Bloque 2: Función principal con parámetros

```python
def main(data_dir: str, img_size: int, batch: int, epochs: int, fine_tune_epochs: int):
    img_shape = (img_size, img_size)  # Tupla (224, 224) para redimensionar imágenes
```
- `data_dir`: Carpeta con las subcarpetas de clases.
- `img_size`: Tamaño de imagen (MobileNetV2 espera 224×224).
- `batch`: Imágenes por lote.
- `epochs`: Épocas con base congelada.
- `fine_tune_epochs`: Épocas con base descongelada.

#### Bloque 3: Carga del dataset

```python
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,                    # Ruta: data/raw/
        validation_split=0.2,        # 20% para validación
        subset="training",           # Este es el 80% de entrenamiento
        seed=1337,                   # Semilla para reproducibilidad
        image_size=img_shape,        # Redimensiona a (224, 224)
        batch_size=batch,            # 32 imágenes por lote
        label_mode="int",            # Labels como enteros: 0, 1, 2...
    )
```

**¿Qué hace `image_dataset_from_directory`?**
1. Lee la estructura de carpetas.
2. Asigna un número a cada subcarpeta (en orden alfabético):
   - `cardboard` → 0
   - `glass` → 1
   - `metal` → 2
   - `paper` → 3
   - `plastic` → 4
   - `trash` → 5
3. Redimensiona cada imagen a 224×224.
4. Agrupa en batches de 32.

```python
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",         # Este es el 20% de validación
        seed=1337,                   # ¡Misma semilla! Para que no se mezclen
        image_size=img_shape,
        batch_size=batch,
        label_mode="int",
    )
```

#### Bloque 4: Guardar metadatos

```python
    class_names = train_ds.class_names  # ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    os.makedirs("model", exist_ok=True)  # Crea carpeta model/ si no existe

    with open("model/labels.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)
        # Guarda: ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

    with open("model/meta.json", "w", encoding="utf-8") as f:
        json.dump({"img_size": img_size, "arch": "MobileNetV2"}, f, indent=2)
        # Guarda: {"img_size": 224, "arch": "MobileNetV2"}
```

**¿Por qué guardar esto?**
- `labels.json`: Para que `app.py` sepa qué significa cada índice.
- `meta.json`: Para que `app.py` redimensione igual que en entrenamiento.

#### Bloque 5: Pipeline de datos optimizado

```python
    AUTOTUNE = tf.data.AUTOTUNE  # TensorFlow decide automáticamente el paralelismo
    train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)
```

| Método     | Qué hace                                                      |
|------------|---------------------------------------------------------------|
| `cache()`  | Guarda en memoria los datos tras la primera lectura           |
| `shuffle(1000)` | Mezcla 1000 elementos para que el modelo no vea patrones de orden |
| `prefetch(AUTOTUNE)` | Prepara el siguiente batch mientras la GPU entrena el actual |

#### Bloque 6: Data Augmentation

```python
    data_augmentation = keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal"),  # Voltea horizontalmente (50% prob)
            keras.layers.RandomRotation(0.05),      # Rota hasta ±18° (0.05 * 360°)
            keras.layers.RandomZoom(0.1),           # Zoom hasta ±10%
        ]
    )
```

**¿Por qué augmentation?**
- El dataset es pequeño (~2,500 imágenes).
- Augmentation genera variaciones artificiales.
- Reduce overfitting: el modelo no memoriza imágenes exactas.

#### Bloque 7: Modelo base (MobileNetV2)

```python
    base = keras.applications.MobileNetV2(
        input_shape=img_shape + (3,),  # (224, 224, 3) — 3 canales RGB
        include_top=False,              # Sin la capa de clasificación original (1000 clases de ImageNet)
        weights="imagenet",             # Pesos preentrenados en ImageNet
    )
    base.trainable = False              # ¡Congelamos! No se actualizan estos pesos (todavía)
```

**¿Qué significa `include_top=False`?**
- MobileNetV2 original termina en una capa Dense de 1000 neuronas (clases de ImageNet).
- Nosotros solo tenemos 6 clases; no nos sirve esa capa.
- `include_top=False` nos da solo el "extractor de características".

#### Bloque 8: Construcción del modelo completo

```python
    inputs = keras.Input(shape=img_shape + (3,))   # Entrada: (224, 224, 3)
    x = data_augmentation(inputs)                   # Aplica augmentation
    x = keras.applications.mobilenet_v2.preprocess_input(x)  # Normaliza a [-1, 1]
    x = base(x, training=False)                     # Pasa por MobileNetV2 (sin entrenar)
    x = keras.layers.GlobalAveragePooling2D()(x)   # Reduce (7, 7, 1280) → (1280,)
    x = keras.layers.Dropout(0.2)(x)               # Apaga 20% de neuronas (regularización)
    outputs = keras.layers.Dense(len(class_names), activation="softmax")(x)  # 6 neuronas
    model = keras.Model(inputs, outputs)
```

**Flujo de datos:**
```
Imagen (224,224,3)
    ↓ Augmentation
Imagen aumentada (224,224,3)
    ↓ preprocess_input (normaliza píxeles de [0,255] a [-1,1])
Tensor normalizado (224,224,3)
    ↓ MobileNetV2 (base congelada)
Features (7,7,1280)
    ↓ GlobalAveragePooling2D
Vector (1280,)
    ↓ Dropout(0.2)
Vector (1280,) con algunas neuronas "apagadas"
    ↓ Dense(6, softmax)
Probabilidades (6,) → [0.1, 0.05, 0.02, 0.03, 0.78, 0.02]
```

#### Bloque 9: Compilación y entrenamiento (fase 1)

```python
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),      # Learning rate = 0.001
        loss="sparse_categorical_crossentropy",      # Pérdida para clasificación
        metrics=["accuracy"],                        # Métrica a monitorear
    )

    print("\n== Entrenamiento (cabeza) ==")
    model.fit(train_ds, validation_data=val_ds, epochs=epochs)  # epochs=5
```

**¿Por qué `Adam`?**
- Adaptativo: ajusta el learning rate por parámetro.
- Robusto: funciona bien sin mucho tuning.

#### Bloque 10: Fine-tuning (fase 2)

```python
    if fine_tune_epochs > 0:
        print("\n== Fine-tuning (descongelar base) ==")
        base.trainable = True  # ¡Ahora sí se actualizan los pesos de MobileNetV2!
        model.compile(
            optimizer=keras.optimizers.Adam(1e-5),  # Learning rate MUCHO más bajo
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.fit(train_ds, validation_data=val_ds, epochs=fine_tune_epochs)  # epochs=3
```

**¿Por qué learning rate 1e-5 en fine-tuning?**
- Los pesos de MobileNetV2 ya están "bien".
- Un learning rate alto (1e-3) los destruiría.
- 1e-5 = ajustes finos, sutiles.

#### Bloque 11: Guardar modelo

```python
    model.save("model/model.keras")
    print("\n Modelo guardado en: model/model.keras")
```

**¿Por qué `.keras` y no `.h5`?**
- `.keras` es el formato nativo de Keras 3 (TensorFlow 2.16+).
- Guarda arquitectura + pesos + configuración del optimizador.
- Más robusto que `.h5` para modelos con capas personalizadas.

#### Bloque 12: CLI (Command Line Interface)

```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/raw")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--fine_tune_epochs", type=int, default=3)
    args = parser.parse_args()

    main(args.data_dir, args.img_size, args.batch, args.epochs, args.fine_tune_epochs)
```

Esto permite ejecutar:
```bash
python train.py --epochs 10 --batch 64
```

---

### 📄 app.py — Bloque por Bloque

#### Bloque 1: Imports

```python
# app.py
import json              # Para leer labels.json y meta.json
import numpy as np       # Para manipular arrays
import streamlit as st   # Framework de la interfaz web
from PIL import Image    # Para abrir y redimensionar imágenes
import tensorflow as tf  # Para cargar y ejecutar el modelo
```

#### Bloque 2: Configuración de página

```python
st.set_page_config(page_title="Recycle CNN", layout="centered")
```

- `page_title`: Título en la pestaña del navegador.
- `layout="centered"`: Contenido centrado (vs. "wide" que usa todo el ancho).

#### Bloque 3: Carga de artefactos (con caché)

```python
@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model("model/model.keras")
    with open("model/labels.json", "r", encoding="utf-8") as f:
        labels = json.load(f)
    with open("model/meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    return model, labels, meta
```

**¿Qué hace `@st.cache_resource`?**
- Streamlit recarga el script en cada interacción del usuario.
- Sin caché, cargaría el modelo (14MB) cada vez → muy lento.
- Con caché, carga una vez y reutiliza en memoria.

```python
model, labels, meta = load_artifacts()
IMG_SIZE = int(meta.get("img_size", 224))
```

- `meta.get("img_size", 224)`: Si no existe la clave, usa 224 por defecto.

#### Bloque 4: Interfaz de usuario

```python
st.title("♻️ Clasificador de Residuos (6 clases)")
st.write("Sube una imagen y el modelo predice: cardboard, glass, metal, paper, plastic, trash.")

uploaded = st.file_uploader("Sube una imagen (JPG/PNG)", type=["jpg", "jpeg", "png"])
```

- `st.title()`: Encabezado grande.
- `st.write()`: Texto normal.
- `st.file_uploader()`: Widget para subir archivos. Retorna `None` si no hay archivo.

#### Bloque 5: Procesamiento de imagen

```python
if uploaded:
    img = Image.open(uploaded).convert("RGB")  # Abre y convierte a RGB (sin canal alpha)
    st.image(img, caption="Imagen subida", use_container_width=True)  # Muestra preview
```

**¿Por qué `.convert("RGB")`?**
- Algunas imágenes PNG tienen 4 canales (RGBA).
- El modelo espera 3 canales (RGB).

```python
    img_resized = img.resize((IMG_SIZE, IMG_SIZE))  # Redimensiona a 224×224
    x = np.array(img_resized, dtype=np.float32)     # Convierte a array numpy
    x = np.expand_dims(x, axis=0)                   # Añade dimensión batch: (1, 224, 224, 3)
```

**¿Por qué `expand_dims`?**
- El modelo espera entrada con forma `(batch, height, width, channels)`.
- Una imagen tiene `(height, width, channels)`.
- `expand_dims` añade la dimensión de batch: `(1, 224, 224, 3)`.

#### Bloque 6: Predicción

```python
    preds = model.predict(x, verbose=0)[0]  # Predicción → (6,) probabilidades
    top = int(np.argmax(preds))             # Índice de la probabilidad más alta
```

- `model.predict(x)` retorna `(1, 6)` (batch de 1, 6 clases).
- `[0]` extrae el primer (y único) elemento: `(6,)`.
- `np.argmax(preds)` retorna el índice del valor máximo.

#### Bloque 7: Mostrar resultados

```python
    st.subheader("Predicción final")
    st.write(f"**{labels[top]}**  —  probabilidad: **{preds[top]:.4f}**")
```

Ejemplo de salida: **plastic** — probabilidad: **0.9234**

```python
    st.subheader("Top 3")
    top3 = sorted(list(enumerate(preds)), key=lambda t: t[1], reverse=True)[:3]
    for i, p in top3:
        st.write(f"- {labels[i]}: {p:.4f}")
```

**Desglose:**
1. `enumerate(preds)` → `[(0, 0.10), (1, 0.04), ..., (4, 0.79), (5, 0.03)]`
2. `sorted(..., reverse=True)` → Ordena de mayor a menor por probabilidad.
3. `[:3]` → Toma los primeros 3.

```python
    st.subheader("Probabilidades (todas las clases)")
    prob_dict = {labels[i]: float(preds[i]) for i in range(len(labels))}
    st.bar_chart(prob_dict)
```

- Crea diccionario: `{"cardboard": 0.10, "glass": 0.04, ...}`.
- `st.bar_chart()` lo grafica.

#### Bloque 8: Estado inicial

```python
else:
    st.info("Sube una imagen para comenzar.")
```

Muestra mensaje azul informativo cuando no hay imagen.

---

## Ejecución Local

### Iniciar la aplicación

```bash
streamlit run app.py
```

### Qué esperar

1. Se abre el navegador automáticamente en `http://localhost:8501`.
2. Sube una imagen de prueba.
3. Observa la predicción, top-3 y gráfico.

### Problemas comunes

| Error                                    | Causa                          | Solución                              |
|------------------------------------------|--------------------------------|---------------------------------------|
| `FileNotFoundError: model/model.keras`   | No has entrenado el modelo     | Ejecuta `python train.py`             |
| `ModuleNotFoundError: No module named X` | Dependencia no instalada       | Ejecuta `pip install -r requirements.txt` |
| Puerto 8501 en uso                       | Otra instancia corriendo       | `streamlit run app.py --server.port 8502` |

---

## Despliegue en Streamlit Cloud

### Prerrequisitos

1. Cuenta en [GitHub](https://github.com).
2. Cuenta en [Streamlit Cloud](https://share.streamlit.io) (gratuita, vinculada a GitHub).
3. Repositorio público (o privado con plan de pago).

### Paso 1: Preparar el repositorio

**Estructura mínima para deploy:**
```
repo/
├── app.py
├── requirements.txt
└── model/
    ├── model.keras
    ├── labels.json
    └── meta.json
```

**Crear `.gitignore`:**
```gitignore
# No subir el dataset (muy pesado)
data/

# No subir entorno virtual
venv/
.venv/

# Archivos del sistema
__pycache__/
*.pyc
.DS_Store
```

### Paso 2: Subir a GitHub

```bash
git init
git add .
git commit -m "Initial commit: Recycle CNN classifier"
git branch -M main
git remote add origin https://github.com/tu-usuario/recycle-cnn.git
git push -u origin main
```

### Paso 3: Configurar Streamlit Cloud

1. Ir a [share.streamlit.io](https://share.streamlit.io).
2. Click en **"New app"**.
3. Seleccionar:
   - **Repository**: `tu-usuario/recycle-cnn`
   - **Branch**: `main`
   - **Main file path**: `app.py`
4. Click en **"Deploy"**.

### Paso 4: Esperar y probar

- El primer deploy tarda 5-10 minutos (instala dependencias).
- La URL será: `https://tu-usuario-recycle-cnn.streamlit.app`.

### Checklist de deploy

- [ ] `requirements.txt` incluye todas las dependencias.
- [ ] `model/model.keras` está commiteado (máx ~100MB).
- [ ] `model/labels.json` y `model/meta.json` están commiteados.
- [ ] `app.py` usa rutas relativas (`model/model.keras`, no `C:\Users\...`).
- [ ] No hay secretos hardcodeados (API keys, contraseñas).

### Problemas frecuentes en deploy

| Problema                          | Causa                                    | Solución                              |
|-----------------------------------|------------------------------------------|---------------------------------------|
| App crash al cargar modelo        | TensorFlow muy pesado para recursos free | Usar versión más ligera o tf-cpu      |
| `FileNotFoundError`               | Archivo no subido a Git                  | Verificar con `git status`            |
| Build timeout                     | Demasiadas dependencias                  | Reducir `requirements.txt`            |
| Modelo >100MB                     | Git LFS necesario                        | Usar Git LFS o comprimir modelo       |
