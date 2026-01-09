# train.py
import argparse
import json
import os
import tensorflow as tf
from tensorflow import keras

def main(data_dir: str, img_size: int, batch: int, epochs: int, fine_tune_epochs: int):
    img_shape = (img_size, img_size)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=1337,
        image_size=img_shape,
        batch_size=batch,
        label_mode="int",
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=1337,
        image_size=img_shape,
        batch_size=batch,
        label_mode="int",
    )

    class_names = train_ds.class_names
    os.makedirs("model", exist_ok=True)

    with open("model/labels.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)

    with open("model/meta.json", "w", encoding="utf-8") as f:
        json.dump({"img_size": img_size, "arch": "MobileNetV2"}, f, indent=2)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)

    data_augmentation = keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.05),
            keras.layers.RandomZoom(0.1),
        ]
    )

    base = keras.applications.MobileNetV2(
        input_shape=img_shape + (3,),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    inputs = keras.Input(shape=img_shape + (3,))
    x = data_augmentation(inputs)
    x = keras.applications.mobilenet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(len(class_names), activation="softmax")(x)
    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print("\n== Entrenamiento (cabeza) ==")
    model.fit(train_ds, validation_data=val_ds, epochs=epochs)

    if fine_tune_epochs > 0:
        print("\n== Fine-tuning (descongelar base) ==")
        base.trainable = True
        model.compile(
            optimizer=keras.optimizers.Adam(1e-5),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.fit(train_ds, validation_data=val_ds, epochs=fine_tune_epochs)

    model.save("model/model.keras")
    print("\n Modelo guardado en: model/model.keras")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/raw")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--fine_tune_epochs", type=int, default=3)
    args = parser.parse_args()

    main(args.data_dir, args.img_size, args.batch, args.epochs, args.fine_tune_epochs)
