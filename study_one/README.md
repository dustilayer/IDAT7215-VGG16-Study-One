# Group 8 · VGG16 on Study One

This VGG16-only experiment uses the exact image assignments in
[the teammate's Study One split](https://github.com/RachelYim67/IDAT7215-3D-Defect-Detection/blob/d6f0c4357c6c4b60bdf0a2c4426e42f15a745f09/results/one/split.csv).
All 271 local originals were checked against the teammate’s SHA-256 inventory.
Train, validation and test contain 189, 27 and 55 images, respectively. The test
set contains 28 defective and 27 non-defective images.

## Reproduce

From the package root, with Python 3.12:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python download_dataset.py
python study_one_vgg16.py --output study_one/reproduced
python summarize_study_one.py study_one/reproduced
```

Keras fetches ImageNet weights on first use. The dataset script downloads the
pinned public original images. Each run writes a fresh output directory.
The included results were generated on macOS ARM with TensorFlow CPU.

## Paired comparisons

| Variant | Input | Pooling |
| --- | --- | --- |
| `paper` | RGB divided by 255, following the authors' example | 4×4 average, stride 4, valid |
| `keras` | Keras VGG16 ImageNet preprocessing | Same 4×4 average |
| `keras_gap` | Same Keras preprocessing | Global average over the full 7×7 feature map |

Compare `paper` → `keras` for preprocessing and `keras` → `keras_gap`
for pooling conditional on Keras preprocessing. The fixed local pool uses only
16 of the 49 spatial positions in the final feature map. Both pooling choices
produce 512 features, so the classifier head has the same parameter count.
Keras preprocessing converts pixel values in [0,255] from RGB to BGR and
subtracts the ImageNet channel means. Each comparison pairs seeds 42, 43
and 44 while holding the image assignments, frozen ImageNet backbone, input
size (224×224), classifier head, optimizer (Adam), learning rate (0.001),
batch size (30), training duration (50 epochs), loss and threshold fixed.
There is no runtime augmentation. Source references include the
[original paper](https://doi.org/10.1016/j.eswa.2024.124293),
[authors' experiment notebook](https://github.com/mahsan2/AM3d/blob/dd1025d7180d91f1128c95817b4d3dd268626b1e/basic_algorithm.ipynb)
and [Keras VGG16 documentation](https://keras.io/api/applications/vgg/vgg_models/).

The export variant is determined by the lowest mean final validation binary
cross-entropy across the three seeds. The selected variant is recorded before
test feature extraction and evaluation. Test metrics describe the same 55
images across seeds. The sample standard deviations describe seed variation
on this fixed split. Forty-eight test images have a same-label training image
within 30 seconds by EXIF DateTime. Performance on independent print runs is
unverified.

## Group fusion

Use `results/fusion/test_predictions.csv` for the three-seed average of the
selected VGG16 variant. `results/fusion_seed42/test_predictions.csv` exports
the fixed first seed if the group uses one trained head per architecture.
Both exports carry `image_id,label,p_defect,prediction` with defect label 1.
In each folder, `teammate_schema.csv` uses the teammate's column names and
class order, where `probability_0` is the defect probability. Join by relative
path and check labels and complete coverage before averaging across models.

## Measured results

Mean ± sample standard deviation across the three paired seeds. Accuracy and
recall are percentages, and the deviations are percentage points.

| Input and pooling | Validation loss | Test accuracy | Defect recall | Misses by seed |
| --- | ---: | ---: | ---: | --- |
| Paper input + local pool | 0.09147 ± 0.00453 | 98.79 ± 1.05 | 97.62 ± 2.06 | 1, 0, 1 |
| Keras input + local pool | 0.05012 ± 0.00597 | 99.39 ± 1.05 | 98.81 ± 2.06 | 1, 0, 0 |
| Keras input + global average pool | 0.00695 ± 0.00185 | 100.00 ± 0.00 | 100.00 ± 0.00 | 0, 0, 0 |

Official preprocessing lowered mean validation loss on this split. With that
preprocessing held fixed, global pooling lowered it further and removed the
remaining miss at the fixed threshold. All three configurations achieved ROC
AUC 1.00 in all three seeds, so the observed gains here concern classification
at threshold 0.5 and validation loss. The three-seed export and the seed-42
export both classified all 55 test images correctly. Full tables are in
`results/summary.csv` and `results/per_seed.csv`.
