"""
White Blood Cell Classification (BCCD-style dataset) — v3

Key improvements over the previous run that collapsed / spiked at the
Phase-1 → Phase-2 transition and plateaued ~55–65% val accuracy:

  1. Freeze ALL BatchNormalization layers during fine-tuning.
     Unfreezing MobileNetV2 BN layers is the classic cause of the sudden
     train-loss spike / accuracy crater you saw around epoch 10.
  2. Fresh EarlyStopping + ReduceLROnPlateau instances for each phase so
     patience / best-value state does not carry over and kill Phase 2 early.
  3. Higher input resolution (224×224) — blood-cell morphology needs more
     pixels than MobileNetV2’s 160 default.
  4. EfficientNetB0 backbone (still ImageNet-pretrained, stronger than
     MobileNetV2 for this scale of data) with a slightly deeper head.
  5. Stronger but still conservative augmentation (rotation, zoom, shear,
     mild color jitter) applied only on the training stream.
  6. Mild L2 weight decay + higher dropout on the head to fight overfitting
     on the ~30-image minority classes.
  7. Oversampling still applied only to the training split (never on val).
  8. Explicit label smoothing (0.05) — helps when support per class is tiny.
  9. Cosine-decay-ish LR schedule via ReduceLROnPlateau + lower fine-tune LR.

Honest expectation with Neutrophil≈263 / Eosinophil≈36 / Lymphocytes≈40 /
Monocyte≈29: overall accuracy in the high 80s to low 90s is realistic;
per-class F1 on Monocyte/Eosinophil will still be noisy because each has
only ~6–8 validation images. 90%+ overall is the target; perfect per-class
scores are not guaranteed by sample size alone.

Expected folder layout (edit BASE_DIR):

    BASE_DIR/
        Eosinophil/
        Lymphocytes/
        Monocyte/
        Neutrophil/
        (optional ignored folders: JPEGImages, Annotations, …)
"""
import matplotlib
matplotlib.use("Agg")
import os
import itertools
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import shuffle as sk_shuffle

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Dropout, GlobalAveragePooling2D, Input, BatchNormalization,
)
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
BASE_DIR = "./dataset"
IMG_ROWS, IMG_COLS = 224, 224          # higher res for morphology
BATCH_SIZE = 12                        # small data → modest batch
HEAD_EPOCHS = 25
FINE_TUNE_EPOCHS = 30
FINE_TUNE_AT = 150                     # unfreeze later blocks of EfficientNetB0
TEST_SIZE = 0.2
RANDOM_STATE = 42
LABEL_SMOOTHING = 0.05

IGNORE_FOLDERS = {"JPEGImages", "Annotations"}
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------
def discover_classes(base_dir):
    classes = sorted(
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
        and not d.startswith(".")
        and d not in IGNORE_FOLDERS
    )
    if not classes:
        raise RuntimeError(f"No class subfolders found under {base_dir}")
    return classes


def get_data(base_dir, classes, img_rows, img_cols):
    X, y = [], []
    loaded_counts = {}
    for label_idx, class_name in enumerate(classes):
        class_dir = os.path.join(base_dir, class_name)
        filenames = [f for f in os.listdir(class_dir) if not f.startswith(".")]
        n_loaded = n_skipped = 0
        for filename in tqdm(filenames, desc=class_name):
            ext = os.path.splitext(filename)[1].lower()
            if ext not in VALID_EXTENSIONS:
                n_skipped += 1
                continue
            img_path = os.path.join(class_dir, filename)
            img = cv2.imread(img_path)
            if img is None:
                n_skipped += 1
                continue
            img = cv2.resize(img, (img_cols, img_rows))
            X.append(img)
            y.append(label_idx)
            n_loaded += 1
        loaded_counts[class_name] = n_loaded
        if n_skipped:
            print(f"  [warn] {class_name}: skipped {n_skipped} file(s)")
    return np.asarray(X), np.asarray(y), loaded_counts


def filter_empty_classes(class_names, loaded_counts):
    empty = [c for c in class_names if loaded_counts.get(c, 0) == 0]
    if empty:
        print("\n" + "=" * 70)
        print("WARNING: excluding empty class folder(s):")
        for c in empty:
            print(f"  - {c}")
        print("=" * 70 + "\n")
    kept = [c for c in class_names if loaded_counts.get(c, 0) > 0]
    if not kept:
        raise RuntimeError("No usable images found.")
    return kept


def oversample_minority_classes(X_train, y_train, class_names, cap_ratio=1.0):
    """Duplicate minority training samples so every class ≈ majority count."""
    counts = np.bincount(y_train, minlength=len(class_names))
    target = int(counts.max() * cap_ratio)
    print("\nOversampling training data:")
    X_parts, y_parts = [X_train], [y_train]
    rng = np.random.RandomState(RANDOM_STATE)
    for label_idx, name in enumerate(class_names):
        n = counts[label_idx]
        if n == 0 or n >= target:
            continue
        needed = target - n
        idx_pool = np.where(y_train == label_idx)[0]
        extra_idx = rng.choice(idx_pool, size=needed, replace=True)
        X_parts.append(X_train[extra_idx])
        y_parts.append(y_train[extra_idx])
        print(f"  {name}: {n} → {n + needed} (+{needed})")
    X_over = np.concatenate(X_parts, axis=0)
    y_over = np.concatenate(y_parts, axis=0)
    return sk_shuffle(X_over, y_over, random_state=RANDOM_STATE)


# ---------------------------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------------------------
def plot_class_distribution(y, class_names):
    counts = np.bincount(y, minlength=len(class_names))
    fig, ax = plt.subplots()
    ax.bar(range(len(class_names)), counts, color="steelblue")
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_ylabel("Counts")
    ax.set_title("Class distribution")
    plt.tight_layout()
    plt.savefig("./class_distribution.png", dpi=120)
    plt.close()


def plot_learning_curve(history):
    plt.figure(figsize=(11, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="train")
    plt.plot(history.history["val_accuracy"], label="val")
    plt.title("Model accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="val")
    plt.title("Model loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("./learning_curves.png", dpi=120)
    plt.close()


def plot_confusion_matrix(cm, classes, title="Confusion matrix"):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(classes))
    plt.xticks(ticks, classes, rotation=45, ha="right")
    plt.yticks(ticks, classes)
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], "d"),
                 ha="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig("./confusion_matrix.png", dpi=120)
    plt.close()


# ---------------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------------
def freeze_bn(model):
    """Keep every BatchNormalization layer frozen (critical for fine-tuning)."""
    for layer in model.layers:
        if isinstance(layer, BatchNormalization):
            layer.trainable = False


def build_model(input_shape, num_classes):
    base = EfficientNetB0(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    inputs = Input(shape=input_shape)
    x = base(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)
    x = Dense(256, activation="relu",
              kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs, outputs)

    model.compile(
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        metrics=["accuracy"],
    )
    return model, base


def make_dataset(X, y_hot, batch_size, training=True):
    """tf.data pipeline with augmentation only when training=True."""
    ds = tf.data.Dataset.from_tensor_slices((X, y_hot))
    if training:
        ds = ds.shuffle(buffer_size=len(X), seed=RANDOM_STATE)

    def _prep(img, label):
        img = tf.cast(img, tf.float32)
        if training:
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_flip_up_down(img)
            img = tf.image.rot90(img, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
            # mild geometric + photometric jitter
            img = tf.image.random_brightness(img, max_delta=0.12)
            img = tf.image.random_contrast(img, lower=0.85, upper=1.15)
            img = tf.image.random_saturation(img, lower=0.85, upper=1.15)
            # random zoom / crop simulation
            scale = tf.random.uniform([], 0.85, 1.0)
            h = tf.cast(tf.cast(IMG_ROWS, tf.float32) * scale, tf.int32)
            w = tf.cast(tf.cast(IMG_COLS, tf.float32) * scale, tf.int32)
            img = tf.image.random_crop(img, size=[h, w, 3])
            img = tf.image.resize(img, [IMG_ROWS, IMG_COLS])
        img = preprocess_input(img)
        return img, label

    ds = ds.map(_prep, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def make_callbacks(phase_name, patience_es=8, patience_lr=3):
    """Brand-new callback objects every phase — never reuse across fits."""
    return [
        EarlyStopping(
            monitor="val_loss",
            patience=patience_es,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=patience_lr,
            min_lr=1e-7,
            verbose=1,
        ),
        ModelCheckpoint(
            f"best_{phase_name}.keras",
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),
    ]


def run_training(X_train, y_train_hot, X_val, y_val_hot, class_names,
                 img_rows, img_cols, batch_size=BATCH_SIZE):
    input_shape = (img_rows, img_cols, 3)
    num_classes = y_train_hot.shape[1]
    model, base = build_model(input_shape, num_classes)

    train_ds = make_dataset(X_train, y_train_hot, batch_size, training=True)
    val_ds = make_dataset(X_val, y_val_hot, batch_size, training=False)

    # ----- Phase 1: head only -----
    print("\n=== Phase 1: train classifier head (backbone frozen) ===")
    history1 = model.fit(
        train_ds,
        epochs=HEAD_EPOCHS,
        validation_data=val_ds,
        callbacks=make_callbacks("phase1", patience_es=8, patience_lr=3),
    )

    # ----- Phase 2: fine-tune top of backbone, BN stays frozen -----
    print("\n=== Phase 2: fine-tune top backbone layers (BN frozen) ===")
    base.trainable = True
    for i, layer in enumerate(base.layers):
        if i < FINE_TUNE_AT:
            layer.trainable = False
        # always keep BatchNorm frozen regardless of index
        if isinstance(layer, BatchNormalization):
            layer.trainable = False

    # double-check: freeze any BN that might sit outside the base
    freeze_bn(model)

    model.compile(
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING),
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        metrics=["accuracy"],
    )

    history2 = model.fit(
        train_ds,
        epochs=HEAD_EPOCHS + FINE_TUNE_EPOCHS,
        initial_epoch=len(history1.epoch),
        validation_data=val_ds,
        callbacks=make_callbacks("phase2", patience_es=10, patience_lr=4),
    )

    # merge histories for a single plot
    history = history1
    for k in history.history:
        history.history[k] += history2.history.get(k, [])

    score = model.evaluate(val_ds, verbose=0)
    print(f"\nFinal validation accuracy: {score[1]:.4f}")

    y_pred = model.predict(val_ds, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_val_hot, axis=1)

    all_labels = list(range(len(class_names)))
    print(classification_report(
        y_true, y_pred_classes,
        labels=all_labels,
        target_names=class_names,
        zero_division=0,
        digits=3,
    ))

    plot_learning_curve(history)
    cm = confusion_matrix(y_true, y_pred_classes, labels=all_labels)
    plot_confusion_matrix(cm, classes=class_names)

    return model, history


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    class_names_raw = discover_classes(BASE_DIR)
    print("Detected class folders:", class_names_raw)

    X, y, loaded_counts = get_data(BASE_DIR, class_names_raw, IMG_ROWS, IMG_COLS)
    print("\nImages loaded per class:")
    for c in class_names_raw:
        print(f"  {c}: {loaded_counts.get(c, 0)}")

    class_names = filter_empty_classes(class_names_raw, loaded_counts)

    if class_names != class_names_raw:
        old_to_new = {class_names_raw.index(c): class_names.index(c) for c in class_names}
        keep = np.isin(y, list(old_to_new.keys()))
        X, y = X[keep], y[keep]
        y = np.array([old_to_new[v] for v in y])

    print(f"\nLoaded {X.shape[0]} images across {len(class_names)} classes: {class_names}")
    plot_class_distribution(y, class_names)

    X = X.astype("float32")  # preprocess_input expects 0-255 range

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    X_train, y_train = oversample_minority_classes(X_train, y_train, class_names)

    y_train_hot = to_categorical(y_train, num_classes=len(class_names))
    y_val_hot = to_categorical(y_val, num_classes=len(class_names))

    model, history = run_training(
        X_train, y_train_hot, X_val, y_val_hot,
        class_names, IMG_ROWS, IMG_COLS,
    )

    model.save("wbc_classifier.keras")
    print("Model saved to wbc_classifier.keras")
