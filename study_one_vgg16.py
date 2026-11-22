"""Run three paired VGG16 ablations using the group's fixed Study One images."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from train_vgg16 import evaluate, write_csv

SEEDS = (42, 43, 44)
VARIANTS = {
    "paper": ("paper", "local"),
    "keras": ("keras", "local"),
    "keras_gap": ("keras", "global"),
}
LABELS = {"badcylinder": 1, "goodcylinder": 0}
BATCH_SIZE = 30
EPOCHS = 50
LEARNING_RATE = 0.001


def read_manifest(path, root):
    with path.open(newline="") as stream:
        records = list(csv.DictReader(stream))
    splits = {split: [] for split in ("train", "validation", "test")}
    for row in records:
        image_id = row["path"]
        folder = row["label"]
        label = LABELS[folder]
        if Path(image_id).parts != (folder, Path(image_id).name) or int(row["class_id"]) != 1 - label:
            raise ValueError(f"Invalid image path or class encoding: {row}")
        splits[row["split"]].append({"image_id": image_id, "label": label})
    expected = {path.relative_to(root).as_posix() for folder in LABELS
                for path in (root / folder).glob("*.JPG")}
    paths = [row["path"] for row in records]
    if len(paths) != 271 or len(set(paths)) != 271 or set(paths) != expected:
        raise ValueError("Study One split must contain every one of the 271 original images exactly once.")
    if [len(rows) for rows in splits.values()] != [189, 27, 55]:
        raise ValueError("Unexpected Study One partition sizes.")
    return splits


def features_for(backbone, root, rows, mode):
    features = {"local": [], "global": []}
    for start in range(0, len(rows), BATCH_SIZE):
        images = []
        for row in rows[start:start + BATCH_SIZE]:
            path = root / row["image_id"]
            pixels = cv2.imread(str(path))
            if pixels is None:
                raise ValueError(f"Unreadable image: {path}")
            rgb = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)
            images.append(cv2.resize(rgb, (224, 224)))
        pixels = np.asarray(images, dtype=np.float32)
        if mode == "paper":
            pixels /= 255.0
        else:
            pixels = tf.keras.applications.vgg16.preprocess_input(pixels)
        local, global_average = backbone(pixels, training=False)
        features["local"].append(local.numpy().reshape(-1, 512))
        features["global"].append(global_average.numpy())
    return {name: np.concatenate(batches) for name, batches in features.items()}


def train_head(x_train, y_train, x_val, y_val, seed):
    tf.keras.utils.set_random_seed(seed)
    head = tf.keras.Sequential([
        tf.keras.Input(shape=(512,)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(2, activation="softmax"),
    ])
    head.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
                 loss="binary_crossentropy", metrics=["accuracy"])
    history = head.fit(x_train, y_train, validation_data=(x_val, y_val),
                       epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0).history
    return head, history


class AveragedHead:
    def __init__(self, heads):
        self.heads = heads

    def predict(self, features, verbose=0):
        scores = [head.predict(features, verbose=verbose) for head in self.heads]
        return np.mean(scores, axis=0, dtype=np.float64)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/dataset"))
    parser.add_argument("--manifest", type=Path, default=Path("study_one/split.csv"))
    parser.add_argument("--output", type=Path, default=Path("study_one/results"))
    args = parser.parse_args()
    splits = read_manifest(args.manifest, args.data)
    args.output.mkdir(parents=True, exist_ok=False)
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.experimental.enable_op_determinism()
    tf.keras.utils.set_random_seed(SEEDS[0])
    base = tf.keras.applications.VGG16(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
    base.trainable = False
    local = tf.keras.layers.AveragePooling2D(4)(base.output)
    gap = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    extractor = tf.keras.Model(base.input, [local, gap])
    inputs = {}
    labels = {split: tf.keras.utils.to_categorical([row["label"] for row in rows], 2)
              for split, rows in splits.items()}
    for mode in ("paper", "keras"):
        for split in ("train", "validation"):
            print(f"Extracting {mode} {split}", flush=True)
            inputs[mode, split] = features_for(extractor, args.data, splits[split], mode)
    fitted = {}
    validation_losses = {}
    for name, (mode, pool) in VARIANTS.items():
        fitted[name] = []
        validation_losses[name] = []
        for seed in SEEDS:
            run = args.output / name / str(seed)
            run.mkdir(parents=True)
            head, history = train_head(inputs[mode, "train"][pool], labels["train"],
                                       inputs[mode, "validation"][pool], labels["validation"], seed)
            head.save(run / "classifier_head.keras")
            write_csv(run / "history.csv", ["epoch", *history],
                      [dict(zip(history, row), epoch=i)
                       for i, row in enumerate(zip(*history.values()), 1)])
            validation_losses[name].append(history["val_loss"][-1])
            fitted[name].append(head)
            print(f"{name} seed={seed}: validation loss={history['val_loss'][-1]:.4f}", flush=True)
    selected = min(VARIANTS, key=lambda name: np.mean(validation_losses[name]))
    metadata = {
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "seeds": SEEDS, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
        "optimizer": "Adam", "learning_rate": LEARNING_RATE, "defect_class": 1,
        "threshold": 0.5, "selection_rule": "lowest mean final validation binary cross-entropy",
        "selected_variant": selected, "validation_losses": validation_losses,
        "tensorflow": tf.__version__, "keras": tf.keras.__version__, "opencv": cv2.__version__,
    }
    (args.output / "configuration.json").write_text(json.dumps(metadata, indent=2) + "\n")
    for mode in ("paper", "keras"):
        print(f"Extracting {mode} test holdout", flush=True)
        inputs[mode, "test"] = features_for(extractor, args.data, splits["test"], mode)
    for name, (mode, pool) in VARIANTS.items():
        for seed, head in zip(SEEDS, fitted[name], strict=True):
            evaluate(head, inputs[mode, "test"][pool], labels["test"], splits["test"],
                     args.output / name / str(seed), "Study One test")
    for identifier, head in [("fusion", AveragedHead(fitted[selected])),
                             ("fusion_seed42", fitted[selected][0])]:
        directory = args.output / identifier
        directory.mkdir()
        mode, pool = VARIANTS[selected]
        evaluate(head, inputs[mode, "test"][pool], labels["test"], splits["test"],
                 directory, "Study One test")
        with (directory / "test_predictions.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        write_csv(directory / "teammate_schema.csv",
                  ["path", "label", "prediction", "probability_0", "probability_1"],
                  [{"path": row["image_id"],
                    "label": "badcylinder" if row["label"] == "1" else "goodcylinder",
                    "prediction": "badcylinder" if row["prediction"] == "1" else "goodcylinder",
                    "probability_0": row["p_defect"],
                    "probability_1": 1 - float(row["p_defect"])} for row in rows])


if __name__ == "__main__":
    main()
