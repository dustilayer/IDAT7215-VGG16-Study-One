# VGG16 Defect Detection for 3D-Printed Cylinders

IDAT7215 Group 8 · Study One · 271 original images

We compared three VGG16 configurations on the group's fixed split. Keras ImageNet preprocessing and global average pooling achieved the lowest mean validation loss. Each of its three training runs correctly classified all 55 test images, including all 28 defective images.

![Mean validation loss and per-seed defect misses for the three VGG16 configurations](figures/vgg16-study-one.png)

## Experimental setup

We used the shared split of 189 training, 27 validation, and 55 test images. Images were resized to 224 × 224. We froze the ImageNet-pretrained VGG16 backbone and trained a 64-unit ReLU head with 0.5 dropout and two-class softmax. Each configuration used Adam, a learning rate of 0.001, a batch size of 30, and 50 epochs at each of seeds 42, 43, and 44. [Shared split](study_one/split.csv) · [Training code](study_one_vgg16.py)

### Step 1. Input preprocessing

The baseline divided RGB pixels by 255, following the authors' example. We then used the VGG16 preprocessing specified by Keras, which converts RGB to BGR and subtracts the ImageNet channel means. The pooling layer and classifier head stayed fixed. Mean final validation loss decreased from 0.09147 to 0.05012. [Authors' notebook](https://github.com/mahsan2/AM3d/blob/dd1025d7180d91f1128c95817b4d3dd268626b1e/basic_algorithm.ipynb) · [Keras documentation](https://keras.io/api/applications/vgg/vgg_models/)

### Step 2. Global average pooling

With Keras preprocessing fixed, we replaced 4 × 4 average pooling with global average pooling. The original layer used stride 4 and valid padding, covering 16 of the 49 spatial positions in the final 7 × 7 feature map. Global average pooling used all 49 positions per channel. Both methods produced 512 features, so the classification head had the same parameter count. Mean final validation loss decreased from 0.05012 to 0.00695.

## Three-seed results

Test accuracy and defect recall are reported as mean ± sample standard deviation in percentage points. The decision threshold was fixed at 0.5.

| Configuration | Mean validation loss | Test accuracy | Defect recall | Missed defects (42 / 43 / 44) |
| --- | ---: | ---: | ---: | ---: |
| Authors' preprocessing + local pool | 0.09147 | 98.79% ± 1.05 | 97.62% ± 2.06 | 1 / 0 / 1 |
| Keras preprocessing + local pool | 0.05012 | 99.39% ± 1.05 | 98.81% ± 2.06 | 1 / 0 / 0 |
| **Keras preprocessing + global average pool** | **0.00695** | **100.00% ± 0.00** | **100.00% ± 0.00** | **0 / 0 / 0** |

The final configuration was selected by the mean final validation loss before test evaluation. All nine runs achieved a test ROC AUC of 1.00. The observed gains were in validation loss and classification at the fixed threshold. [Per-seed metrics](study_one/results/per_seed.csv) · [Confusion matrix](study_one/results/fusion/confusion_matrix.png)

**Scope of 100% accuracy.** The result applies to the 55 images in this fixed Study One test split. The three seeds measure training variation on the same split. In the image timestamps, 48 test images had a same-class training image within 30 seconds. Nearby frames may make this task easier than testing on unseen prints. Performance on independent print runs remains unverified.

## Reproduction and group fusion

Follow the [reproduction instructions](study_one/README.md) to download the pinned original images and rerun the nine experiments. The repository includes the trained classifier heads, learning curves, per-image predictions, and experiment configuration.

Use the [three-seed prediction CSV](study_one/results/fusion/teammate_schema.csv) for group fusion. Join by `path` and average `probability_0`, the probability of a defect in the shared export format. All 55 paths and labels were verified against the teammate's Study One predictions. A [fixed seed 42 export](study_one/results/fusion_seed42/teammate_schema.csv) is available for comparisons that use one trained head per architecture.
