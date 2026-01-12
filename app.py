# app.py
import json
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf

st.set_page_config(page_title="Recycle CNN", layout="centered")

@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model("model/model.keras")
    with open("model/labels.json", "r", encoding="utf-8") as f:
        labels = json.load(f)
    with open("model/meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    return model, labels, meta

model, labels, meta = load_artifacts()
IMG_SIZE = int(meta.get("img_size", 224))

st.title("Clasificador de Residuos con CNN")
st.write("Sube una imagen y el modelo predice: carton (cardboard), vidrio (glass), metal, papel (paper), plástico (plastic), basura (trash).")

uploaded = st.file_uploader("Sube una imagen (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Imagen subida", use_container_width=True)

    img_resized = img.resize((IMG_SIZE, IMG_SIZE))
    x = np.array(img_resized, dtype=np.float32)
    x = np.expand_dims(x, axis=0)  # (1, H, W, 3)

    preds = model.predict(x, verbose=0)[0]
    top = int(np.argmax(preds))

    st.subheader("Predicción final")
    st.write(f"**{labels[top]}**  —  probabilidad: **{preds[top]:.4f}**")

    st.subheader("Top 3")
    top3 = sorted(list(enumerate(preds)), key=lambda t: t[1], reverse=True)[:3]
    for i, p in top3:
        st.write(f"- {labels[i]}: {p:.4f}")

    st.subheader("Probabilidades (todas las clases)")
    prob_dict = {labels[i]: float(preds[i]) for i in range(len(labels))}
    st.bar_chart(prob_dict)
else:
    st.info("Sube una imagen para comenzar.")
