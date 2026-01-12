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

# Carga del modelo y datos necesarios
model = keras.models.load_model("model/model.keras")

with open("model/labels.json", "r") as f:
    class_names = json.load(f)

with open("model/meta.json", "r") as f:
    meta = json.load(f)
    img_size = meta["img_size"]

# Cargar dataset de validación (20% de entrenamiento)
val_ds = tf.keras.utils.image_dataset_from_directory(
    "data/raw",
    validation_split=0.2,
    subset="validation",
    seed=1337,  # Mismo seed que en train.py
    image_size=(img_size, img_size),
    batch_size=32,
    label_mode="int",
    shuffle=False  
)

# Obtención de predicciones
y_true = []  # Etiquetas reales
y_pred = []  # Etiquetas predichas

for images, labels in val_ds:
    predictions = model.predict(images, verbose=0)
    y_pred.extend(np.argmax(predictions, axis=1))
    y_true.extend(labels.numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Cálculo de métricas

# Accuracy general
accuracy = np.mean(y_true == y_pred)
print(f"\nAccuracy General: {accuracy:.4f} ({accuracy*100:.2f}%)\n")

# Classification Report (Precision, Recall, F1 por clase)
print("=" * 50)
print("CLASSIFICATION REPORT")
print("=" * 50)
report = classification_report(y_true, y_pred, target_names=class_names)
print(report)

# Guardar reporte en archivo
with open("model/classification_report.txt", "w") as f:
    f.write(f"Accuracy: {accuracy:.4f}\n\n")
    f.write(report)
print("Reporte guardado en: model/classification_report.txt")

# Matriz de Confusión
print("\n" + "=" * 50)
print("CONFUSION MATRIX")
print("=" * 50)

cm = confusion_matrix(y_true, y_pred)
print(cm)

# Visualización de la matriz
fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(ax=ax, cmap="Blues", values_format="d")
plt.title("Matriz de Confusión - Clasificador de Residuos")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("model/confusion_matrix.png", dpi=150)
plt.show()
print("Matriz guardada en: model/confusion_matrix.png")

# Métricas en formato JSON
report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

# Guardar métricas en JSON
with open("model/metrics.json", "w") as f:
    json.dump(report_dict, f, indent=2)
print("Métricas JSON guardadas en: model/metrics.json")