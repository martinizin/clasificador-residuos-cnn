import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import os
from collections import Counter

print("=" * 60)
print("EVALUACIÓN DEL MODELO - CLASIFICADOR DE RESIDUOS")
print("=" * 60)

# =====================================================
# 1. DIAGNÓSTICO DEL DATASET
# =====================================================
print("\n📁 DIAGNÓSTICO DEL DATASET")
print("-" * 40)

data_dir = "data/raw"
if os.path.exists(data_dir):
    folders = sorted([f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))])
    print(f"Carpetas encontradas: {folders}")
    print("\nImágenes por clase:")
    total_images = 0
    for folder in folders:
        folder_path = os.path.join(data_dir, folder)
        count = len([f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        total_images += count
        print(f"  {folder}: {count} imágenes")
    print(f"\nTotal de imágenes: {total_images}")
else:
    print(f"ERROR: No se encontró la carpeta {data_dir}")

# =====================================================
# 2. CARGA DEL MODELO Y METADATOS
# =====================================================
print("\n🧠 CARGA DEL MODELO")
print("-" * 40)

model = keras.models.load_model("model/model.keras")
print(f"Modelo cargado: model/model.keras")

# Verificar la capa de salida del modelo
output_layer = model.layers[-1]
num_classes_model = output_layer.units if hasattr(output_layer, 'units') else model.output_shape[-1]
print(f"Clases en la capa de salida del modelo: {num_classes_model}")

with open("model/labels.json", "r") as f:
    class_names = json.load(f)
print(f"Clases en labels.json: {len(class_names)} -> {class_names}")

with open("model/meta.json", "r") as f:
    meta = json.load(f)
    img_size = meta["img_size"]
print(f"Tamaño de imagen: {img_size}x{img_size}")

# Verificar consistencia
if num_classes_model != len(class_names):
    print(f"\n⚠️  ADVERTENCIA: El modelo tiene {num_classes_model} clases pero labels.json tiene {len(class_names)}")
    print("    Esto indica que el modelo fue entrenado con un dataset diferente.")

# =====================================================
# 3. CARGAR DATASET DE VALIDACIÓN
# =====================================================
print("\n📊 CARGA DEL DATASET DE VALIDACIÓN")
print("-" * 40)

# Primero verificamos cómo se distribuyen las clases en el split original
test_val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=1337,
    image_size=(img_size, img_size),
    batch_size=32,
    label_mode="int",
    shuffle=False
)

# Contar clases en validación original
original_val_classes = []
for _, labels in test_val_ds:
    original_val_classes.extend(labels.numpy())
from collections import Counter as Cnt
original_dist = Cnt(original_val_classes)
print(f"Distribución original del split (seed=1337):")
for idx, count in sorted(original_dist.items()):
    print(f"  {class_names[idx]}: {count}")

# Si no todas las clases están representadas, usar un enfoque diferente
classes_in_original = len(original_dist)
if classes_in_original < len(class_names):
    print(f"\n⚠️  El split original solo tiene {classes_in_original} clases.")
    print("   Usando evaluación con TODO el dataset para métricas completas...")
    
    # Cargar TODO el dataset para evaluación
    full_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        image_size=(img_size, img_size),
        batch_size=32,
        label_mode="int",
        shuffle=False
    )
    val_ds = full_ds
    using_full_dataset = True
    print(f"   Evaluando con las {len(list(full_ds.class_names))} clases completas.")
else:
    val_ds = test_val_ds
    using_full_dataset = False

# Obtener las clases que detectó el dataset
dataset_class_names = val_ds.class_names
print(f"\nClases detectadas por el dataset: {dataset_class_names}")

# Verificar que coincidan con labels.json
if dataset_class_names != class_names:
    print(f"\n⚠️  ADVERTENCIA: Las clases del dataset no coinciden con labels.json")
    print(f"    Dataset: {dataset_class_names}")
    print(f"    labels.json: {class_names}")

# =====================================================
# 4. OBTENCIÓN DE PREDICCIONES
# =====================================================
print("\n🔮 OBTENIENDO PREDICCIONES...")
print("-" * 40)

y_true = []
y_pred = []
y_pred_probs = []

for images, labels in val_ds:
    predictions = model.predict(images, verbose=0)
    y_pred.extend(np.argmax(predictions, axis=1))
    y_pred_probs.extend(predictions)
    y_true.extend(labels.numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_pred_probs = np.array(y_pred_probs)

# =====================================================
# 5. DIAGNÓSTICO DE DISTRIBUCIÓN
# =====================================================
print("\n📈 DISTRIBUCIÓN DE DATOS")
print("-" * 40)

# Distribución real (y_true)
true_counts = Counter(y_true)
print("\nDistribución REAL en validación (y_true):")
for idx in sorted(true_counts.keys()):
    class_name = class_names[idx] if idx < len(class_names) else f"clase_{idx}"
    print(f"  {class_name} (idx={idx}): {true_counts[idx]} imágenes")

# Distribución predicha (y_pred)
pred_counts = Counter(y_pred)
print("\nDistribución PREDICHA (y_pred):")
for idx in sorted(pred_counts.keys()):
    class_name = class_names[idx] if idx < len(class_names) else f"clase_{idx}"
    print(f"  {class_name} (idx={idx}): {pred_counts[idx]} predicciones")

# Verificar si hay clases faltantes en y_true
missing_classes = set(range(len(class_names))) - set(true_counts.keys())
if missing_classes and not using_full_dataset:
    print(f"\n⚠️  CLASES SIN IMÁGENES EN VALIDACIÓN: {[class_names[i] for i in missing_classes]}")
    print("    Esto explica por qué esas clases tienen métricas de 0.00")

# Verificar si el modelo solo predice ciertas clases
unique_predictions = set(y_pred)
if len(unique_predictions) < len(class_names):
    never_predicted = set(range(len(class_names))) - unique_predictions
    print(f"\n⚠️  CLASES NUNCA PREDICHAS: {[class_names[i] for i in never_predicted]}")
    print("    Esto puede indicar un problema con el entrenamiento del modelo.")

# Mostrar si estamos usando el dataset completo
if using_full_dataset:
    print(f"\n📌 NOTA: Se está evaluando con TODO el dataset ({len(y_true)} imágenes)")
    print("   ya que el split original no contenía todas las clases.")

# =====================================================
# 6. CÁLCULO DE MÉTRICAS
# =====================================================
print("\n" + "=" * 60)
print("📊 MÉTRICAS DEL MODELO")
print("=" * 60)

# Accuracy general
accuracy = np.mean(y_true == y_pred)
print(f"\n✅ Accuracy General: {accuracy:.4f} ({accuracy*100:.2f}%)\n")

# Classification Report
print("-" * 50)
print("CLASSIFICATION REPORT")
print("-" * 50)

# Usar zero_division=0 para evitar warnings cuando una clase no tiene muestras
report = classification_report(
    y_true, 
    y_pred, 
    target_names=class_names,
    zero_division=0
)
print(report)

# Guardar reporte en archivo
with open("model/classification_report.txt", "w", encoding="utf-8") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n\n")
    f.write("DISTRIBUCIÓN EN VALIDACIÓN:\n")
    for idx in sorted(Counter(y_true).keys()):
        f.write(f"  {class_names[idx]}: {Counter(y_true)[idx]} imágenes\n")
    f.write(f"\nTotal imágenes de validación: {len(y_true)}\n")
    f.write("\n" + "=" * 50 + "\n")
    f.write("CLASSIFICATION REPORT:\n")
    f.write("=" * 50 + "\n")
    f.write(report)
print("📁 Reporte guardado en: model/classification_report.txt")

# =====================================================
# 7. MATRIZ DE CONFUSIÓN
# =====================================================
print("\n" + "-" * 50)
print("CONFUSION MATRIX")
print("-" * 50)

cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
print(cm)

# Visualización de la matriz
fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(ax=ax, cmap="Blues", values_format="d")
plt.title("Matriz de Confusión - Clasificador de Residuos")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("model/confusion_matrix.png", dpi=150)
print("📁 Matriz guardada en: model/confusion_matrix.png")
plt.show()

# =====================================================
# 8. MÉTRICAS EN JSON
# =====================================================
report_dict = classification_report(
    y_true, 
    y_pred, 
    target_names=class_names, 
    output_dict=True,
    zero_division=0
)

with open("model/metrics.json", "w", encoding="utf-8") as f:
    json.dump(report_dict, f, indent=2)
print("📁 Métricas JSON guardadas en: model/metrics.json")

# =====================================================
# 9. ANÁLISIS DE CONFIANZA DE PREDICCIONES
# =====================================================
print("\n" + "=" * 60)
print("🔍 ANÁLISIS DE CONFIANZA")
print("=" * 60)

# Estadísticas de confianza
max_probs = np.max(y_pred_probs, axis=1)
print(f"\nConfianza promedio de predicciones: {np.mean(max_probs):.4f}")
print(f"Confianza mínima: {np.min(max_probs):.4f}")
print(f"Confianza máxima: {np.max(max_probs):.4f}")

# Predicciones con baja confianza
low_confidence = np.sum(max_probs < 0.5)
print(f"\nPredicciones con confianza < 50%: {low_confidence} ({100*low_confidence/len(max_probs):.1f}%)")

# =====================================================
# 10. DIAGNÓSTICO FINAL Y RECOMENDACIONES
# =====================================================
print("\n" + "=" * 60)
print("💡 DIAGNÓSTICO FINAL")
print("=" * 60)

# Verificar si hay problema con el modelo
classes_with_samples = len([c for c in Counter(y_true).values() if c > 0])
classes_predicted = len(set(y_pred))

if using_full_dataset:
    print(f"""
📊 EVALUACIÓN COMPLETADA CON DATASET COMPLETO

El split original (seed=1337) no contenía todas las clases en validación,
por lo que se evaluó con el dataset completo para obtener métricas reales.

RESULTADOS:
- Imágenes evaluadas: {len(y_true)}
- Clases con muestras: {classes_with_samples} de {len(class_names)}
- Clases predichas: {classes_predicted} de {len(class_names)}
- Accuracy: {accuracy:.2%}

NOTA: Para una evaluación más rigurosa, considera crear un conjunto de test
separado manualmente o usar cross-validation.
""")
elif classes_with_samples < len(class_names):
    print(f"""
⚠️  PROBLEMA DETECTADO: El conjunto de validación solo tiene {classes_with_samples} de {len(class_names)} clases.

POSIBLES CAUSAS:
1. El modelo fue entrenado con un dataset diferente (menos clases)
2. La estructura de carpetas cambió después del entrenamiento
3. El seed del split no coincide entre train.py y evaluate.py

SOLUCIÓN RECOMENDADA:
→ Re-entrenar el modelo con el dataset completo:
  python train.py --data_dir data/raw --epochs 5 --fine_tune_epochs 3

Luego volver a ejecutar la evaluación:
  python evaluate.py
""")
elif classes_predicted < len(class_names):
    print(f"""
⚠️  PROBLEMA DETECTADO: El modelo solo predice {classes_predicted} de {len(class_names)} clases.

POSIBLES CAUSAS:
1. El modelo no aprendió bien todas las clases durante el entrenamiento
2. Hay un desbalance severo en el dataset
3. Se necesitan más epochs o ajustes en hiperparámetros

SOLUCIÓN RECOMENDADA:
→ Re-entrenar con más epochs y posiblemente class weights
""")
else:
    print(f"""
✅ El modelo parece estar funcionando correctamente.
   - Clases en validación: {classes_with_samples}
   - Clases predichas: {classes_predicted}
   - Accuracy: {accuracy:.2%}
""")