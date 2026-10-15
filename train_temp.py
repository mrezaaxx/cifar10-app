"""Latih model CNN sementara untuk CIFAR-10 agar aplikasi bisa diuji.

Arsitektur mengikuti notebook tim (Conv 32/64/128 + Dense 128 + Dropout),
ditambah BatchNormalization, Dropout per blok, dan augmentasi ringan.
Ganti model/model.keras dengan model final tim bila sudah tersedia.

Jalankan:  python train_temp.py [--epochs 15]
"""
import argparse
import io
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from tensorflow.keras import layers, models

ROOT = Path(__file__).parent
EPOCHS_DEFAULT = 15
# Mirror CIFAR-10 di Hugging Face; server asli (cs.toronto.edu) sangat lambat.
HF_URL = "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/{}-00000-of-00001.parquet"


def load_split(split):
    """Muat CIFAR-10 dari parquet HF -> (x uint8 [N,32,32,3], y int [N])."""
    path = ROOT / "data" / f"{split}.parquet"
    if not path.exists():
        path.parent.mkdir(exist_ok=True)
        print(f"Mengunduh {split} ...")
        urllib.request.urlretrieve(HF_URL.format(split), path)
    df = pd.read_parquet(path)
    x = np.stack([np.asarray(Image.open(io.BytesIO(d["bytes"])).convert("RGB")) for d in df["img"]])
    y = df["label"].to_numpy()
    return x, y


def build_model(num_classes=10):
    augment = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomTranslation(0.1, 0.1),
        ],
        name="augmentation",
    )

    def conv_block(x, filters):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D()(x)
        return layers.Dropout(0.25)(x)

    inputs = layers.Input(shape=(32, 32, 3))
    x = augment(inputs)  # hanya aktif saat training
    x = conv_block(x, 32)
    x = conv_block(x, 64)
    x = conv_block(x, 128)
    x = layers.Flatten()(x)
    x = layers.Dense(128, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs, outputs, name="cifar10_cnn_temp")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS_DEFAULT)
    args = parser.parse_args()

    x_train, y_train = load_split("train")
    x_test, y_test = load_split("test")
    # Normalisasi HARUS sama dengan app.py (÷255).
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    model = build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=4, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
    ]
    history = model.fit(
        x_train,
        y_train,
        validation_split=0.1,
        epochs=args.epochs,
        batch_size=64,
        callbacks=callbacks,
        verbose=2,
    )

    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {acc:.4f} | Test loss: {loss:.4f}")

    out = ROOT / "model" / "model.keras"
    out.parent.mkdir(exist_ok=True)
    model.save(out)
    print(f"Model disimpan ke {out}")

    metrics = {
        "test_accuracy": round(float(acc), 4),
        "test_loss": round(float(loss), 4),
        "epochs_run": len(history.history["loss"]),
        "tensorflow_version": tf.__version__,
    }
    (ROOT / "model" / "metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
