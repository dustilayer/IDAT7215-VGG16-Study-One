"""Train and evaluate the VGG16 baseline on the public AMdataset images."""

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, average_precision_score,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split

LABELS = {"goodcylinder": 0, "badcylinder": 1}
SEED = 42


def make_split(root, protocol):
    rows = [
        {"image_id": path.relative_to(root).as_posix(), "label": label}
        for folder, label in LABELS.items()
        for path in sorted((root / folder).iterdir())
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    if {row["label"] for row in rows} != {0, 1}:
        raise ValueError("Expected images in both goodcylinder and badcylinder.")
    if protocol == "chronological":
        return chronological_split(root, rows)
    train, holdout = train_test_split(
        rows, test_size=0.30, random_state=SEED,
        stratify=[row["label"] for row in rows],
    )
    validation, test = train_test_split(
        holdout, test_size=0.66, random_state=SEED,
        stratify=[row["label"] for row in holdout],
    )
    return {"train": train, "validation": validation, "test": test}


def chronological_split(root, rows):
    """Hold out later frames within each label and leave three-minute gaps."""
    splits = {"train": [], "validation": [], "test": []}
    gap = timedelta(minutes=3)
    for label in LABELS.values():
        dated_rows = []
        for row in rows:
            if row["label"] == label:
                with Image.open(root / row["image_id"]) as image:
                    timestamp = datetime.strptime(image.getexif()[306], "%Y:%m:%d %H:%M:%S")
                dated_rows.append((timestamp, row["image_id"], row))
        dated_rows.sort()
        validation_start = dated_rows[int(0.65 * len(dated_rows))][0]
        test_start = dated_rows[int(0.80 * len(dated_rows))][0]
        for timestamp, _, row in dated_rows:
            if timestamp <= validation_start - gap:
                splits["train"].append(row)
            elif validation_start <= timestamp <= test_start - gap:
                splits["validation"].append(row)
            elif timestamp >= test_start:
                splits["test"].append(row)
    if any({row["label"] for row in rows} != {0, 1} for rows in splits.values()):
        raise ValueError("Chronological split requires both labels in every partition.")
    return splits


def build_extractor():
    backbone = tf.keras.applications.VGG16(
        include_top=False, weights="imagenet", input_shape=(224, 224, 3),
    )
    backbone.trainable = False
    pooled = tf.keras.layers.AveragePooling2D(pool_size=4)(backbone.output)
    return tf.keras.Model(backbone.input, pooled)


def extract_features(paths, extractor, batch_size):
    features = []
    for start in range(0, len(paths), batch_size):
        images = []
        for path in paths[start:start + batch_size]:
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"Could not decode {path}")
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(cv2.resize(rgb, (224, 224)))
        batch = np.asarray(images, dtype=np.float32) / 255.0
        features.append(extractor(batch, training=False).numpy())
    return np.concatenate(features)


def write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def evaluate(head, features, labels, rows, output, protocol):
    probabilities = head.predict(features, verbose=0)[:, 1]
    truth = labels.argmax(axis=1)
    predicted = (probabilities > 0.5).astype(int)
    write_csv(
        output / "test_predictions.csv",
        ["image_id", "label", "p_defect", "prediction"],
        [dict(row, p_defect=float(score), prediction=int(pred))
         for row, score, pred in zip(rows, probabilities, predicted, strict=True)],
    )
    matrix = confusion_matrix(truth, predicted, labels=[0, 1])
    metrics = {
        "n_test": len(rows),
        "accuracy": accuracy_score(truth, predicted),
        "defect_precision": precision_score(truth, predicted, zero_division=0),
        "defect_recall": recall_score(truth, predicted, zero_division=0),
        "defect_f1": f1_score(truth, predicted, zero_division=0),
        "specificity": recall_score(truth, predicted, pos_label=0),
        "roc_auc": roc_auc_score(truth, probabilities),
        "average_precision": average_precision_score(truth, probabilities),
        "confusion_matrix": matrix.tolist(),
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    display = ConfusionMatrixDisplay(matrix, display_labels=["No defect", "Defect"])
    display.plot(cmap="Blues", colorbar=False)
    display.ax_.set_title(f"VGG16 · {protocol.capitalize()} holdout")
    display.figure_.savefig(output / "confusion_matrix.png", dpi=180, bbox_inches="tight")
    plt.close(display.figure_)
    print(json.dumps(metrics, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/dataset"))
    parser.add_argument("--output", type=Path, default=Path("results/adam"))
    parser.add_argument("--optimizer", choices=["adam", "rmsprop"], default="adam")
    parser.add_argument("--split-protocol", choices=["stratified", "chronological"], default="stratified")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=30)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch size must be positive")

    tf.keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    splits = make_split(args.data, args.split_protocol)
    args.output.mkdir(parents=True, exist_ok=False)
    configuration = {
        "seed": SEED, "optimizer": args.optimizer, "epochs": args.epochs,
        "batch_size": args.batch_size, "learning_rate": 0.001,
        "tensorflow_version": tf.__version__, "opencv_version": cv2.__version__,
        "positive_label": "badcylinder", "split_protocol": args.split_protocol,
        "decision_threshold": 0.5,
    }
    (args.output / "configuration.json").write_text(
        json.dumps(configuration, indent=2) + "\n"
    )
    write_csv(
        args.output / "split.csv", ["image_id", "label", "split"],
        [dict(row, split=split) for split, rows in splits.items() for row in rows],
    )

    extractor = build_extractor()

    # With a frozen backbone and no runtime augmentation, extract once.
    inputs = {}
    for split, rows in splits.items():
        print(f"{split}: {len(rows)} images, labels {dict(Counter(r['label'] for r in rows))}", flush=True)
        paths = [args.data / row["image_id"] for row in rows]
        features = extract_features(paths, extractor, args.batch_size)
        labels = tf.keras.utils.to_categorical([row["label"] for row in rows], 2)
        inputs[split] = (features, labels)

    head = tf.keras.Sequential([
        tf.keras.Input(shape=extractor.output_shape[1:]),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(2, activation="softmax"),
    ])
    optimizer_class = {"adam": tf.keras.optimizers.Adam, "rmsprop": tf.keras.optimizers.RMSprop}
    head.compile(
        optimizer=optimizer_class[args.optimizer](learning_rate=0.001),
        loss="binary_crossentropy", metrics=["accuracy"],
    )
    history = head.fit(
        *inputs["train"], validation_data=inputs["validation"],
        epochs=args.epochs, batch_size=args.batch_size, verbose=2,
    ).history
    write_csv(
        args.output / "history.csv", ["epoch", *history],
        [dict(zip(history, values), epoch=index)
         for index, values in enumerate(zip(*history.values()), 1)],
    )
    figure, axes = plt.subplots(1, 2, figsize=(8, 3), layout="constrained")
    for axis, metric in zip(axes, ["loss", "accuracy"]):
        epochs = range(1, args.epochs + 1)
        axis.plot(epochs, history[metric], label="Train")
        axis.plot(epochs, history[f"val_{metric}"], label="Validation")
        axis.set(xlabel="Epoch", ylabel=metric.capitalize())
        axis.legend()
    figure.savefig(args.output / "learning_curves.png", dpi=180)
    plt.close(figure)
    head.save(args.output / "classifier_head.keras")
    evaluate(head, *inputs["test"], splits["test"], args.output, args.split_protocol)


if __name__ == "__main__":
    main()
