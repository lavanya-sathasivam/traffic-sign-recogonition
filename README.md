# Real-Time Traffic Sign Recognition Using CNN and Image Augmentation

## 1. Project Title

**Real-Time Traffic Sign Recognition Using CNN and Image Augmentation**

## 2. Project Overview

This project builds an image-classification system that recognizes traffic signs from cropped sign images using a custom Convolutional Neural Network (CNN).

The overall idea is:

```text
traffic sign image -> preprocessing -> CNN -> predicted class -> confidence
```

1. A traffic-sign image is loaded.
2. It is converted to RGB, resized to `32x32`, and pixel values are normalized to `[0, 1]`.
3. The custom CNN processes the `32x32x3` input.
4. The output layer produces probabilities for 43 classes with Softmax.
5. The class with the highest probability is returned as the predicted class ID.
6. That highest probability is reported as the confidence.

Training uses augmented training images to improve robustness. Validation is used to monitor training. A separate test set is used only once for final evaluation. A prediction notebook allows a user to upload an image and get a prediction from the saved model.

## 3. Problem Statement

Traffic signs must be recognized correctly for driver assistance and autonomous driving. Manual recognition does not scale, and road images vary in lighting, size, angle, background, and image quality.

The problem addressed here is:

> Given a cropped traffic-sign image, automatically predict which of 43 traffic-sign classes it belongs to.

This project does **not** detect or localize signs in full road scenes. It classifies already-cropped sign images. Live webcam/video inference is also not implemented; the current demo predicts on uploaded images.

## 4. Objectives

No separate objectives document was found in the repository. The following objectives are derived directly from the implemented notebook pipeline:

1. Explore the provided traffic-sign dataset and document classes, image properties, imbalance, and file quality.
2. Prepare fixed-size, normalized train/validation inputs while keeping the test set untouched.
3. Apply training-only image augmentation suitable for traffic signs.
4. Train and save a custom CNN for 43-class traffic-sign classification.
5. Evaluate the saved model on the untouched test set with accuracy and class-aware metrics.
6. Provide a notebook demo that predicts class ID and confidence for an uploaded image.

## 5. Dataset

### 5.1 Dataset name/type

The repository contains a local labeled traffic-sign image dataset with a GTSRB-like layout:

- `dataset/Train/`
- `dataset/Test/`
- `dataset/Meta/`
- `dataset/Train.csv`
- `dataset/Test.csv`
- `dataset/Meta.csv`

The exploration notebook states that the local files strongly resemble the published GTSRB CSV/image schema, but provenance is not asserted from folder structure alone. Therefore:

- Dataset provenance/publication name: **Not available** beyond the local files.
- Dataset type: **labeled multi-class traffic-sign image dataset with CSV metadata and PNG images.**

### 5.2 Verified dataset summary

| Item | Verified value |
|---|---|
| Number of classes | 43 |
| Class IDs | 0–42, continuous; same set in Train, Test, and Meta |
| Training images | 39,209 |
| Test images | 12,630 |
| Train image organization | `Train/<ClassId>/*.png`, with 43 numeric folders `0`–`42` |
| Test image organization | Flat folder `Test/*.png`, e.g. `Test/00000.png` |
| Image format | PNG |
| Color mode/channels | RGB, 3 channels for all 51,839 inspected train/test references |
| Original image dimensions | Width 25–266, Height 25–232; smallest `25x25`, largest `266x232` |
| Pixel values | 0–255 before normalization |
| Train labels source | `Train.csv` |
| Test labels source | `Test.csv` |
| `GT-final_test.csv` | Present under `Test/` but empty, size 0 bytes |
| Human-readable class names | Not provided by the dataset |
| Missing referenced images | 0 train, 0 test |
| Unreadable/corrupted referenced images | 0 train, 0 test |
| `Meta/` contents | 43 class images `0.png`–`42.png` plus two lock files; no class-name column |

CSV columns verified from the exploration notebook:

- `Train.csv`: `(39209, 8)` — `Width, Height, Roi.X1, Roi.Y1, Roi.X2, Roi.Y2, ClassId, Path`
- `Test.csv`: `(12630, 8)` — same columns
- `Meta.csv`: `(43, 5)` — `Path, ClassId, ShapeId, ColorId, SignId`

`Meta.csv` provides `ShapeId`, `ColorId`, and `SignId` mappings. It has no column containing human-readable names or labels. Example train path: `Train/20/00020_00000_00000.png`. Example test path: `Test/00000.png`.

The project uses the full images resized to `32x32`. ROI columns are present in the CSVs but ROI cropping is not applied in the preparation/training notebooks.

## 6. Dataset Exploration

Exploration was performed in:

```text
notebooks/01_dataset_exploration.ipynb
```

The notebook checked:

- Class distribution in train and test.
- Image dimensions and dimension histograms.
- Image format, color mode, channels, and pixel range.
- Corrupted/unreadable images.
- Missing files referenced by the CSVs.
- Train/test/Meta folder organization.
- Duplicate paths.
- Files not referenced by metadata.
- Representative images across classes and multiple examples for selected classes.

Important observations:

- **Class imbalance:** training images per class range from 210 to 2,250, mean 911.84, median 600.00.
- Significantly underrepresented classes by the notebook threshold, less than half the mean: `0, 6, 16, 19, 20, 21, 22, 24, 27, 29, 30, 32, 34, 36, 37, 39, 40, 41, 42`.
- Significantly overrepresented classes, more than 1.5x the mean: `1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 25, 38`.
- Most common original sizes are around `29x29` to `40x40`, but sizes vary widely, so resizing is required for a fixed CNN input.
- All 51,839 CSV-referenced train/test images were readable RGB PNGs.
- Filesystem checks found 39,209 train image files, 12,630 test `.png` files plus empty `GT-final_test.csv`, and no empty numeric train folders.
- No PNG files were unreferenced by the three metadata CSVs, and no duplicate Train/Test paths were found.

## 7. Data Preparation

Data preparation was performed in:

```text
notebooks/02_data_preparation.ipynb
```

Actual preprocessing:

| Step | Value |
|---|---|
| Split method | Stratified `train_test_split` |
| Validation fraction | 20% |
| `random_state` | 42 |
| Training samples | 31,367 |
| Validation samples | 7,842 |
| Classes in train | All 43, classes 0–42 |
| Classes in validation | All 43, classes 0–42 |
| Resize | `32x32` with `PIL.Image.Resampling.LANCZOS` |
| Color | Converted to RGB, 3 channels |
| Normalization | `pixel / 255.0`, float32, range `[0, 1]` |
| Final shapes | `X_train (31367, 32, 32, 3)`, `y_train (31367,)`, `X_val (7842, 32, 32, 3)`, `y_val (7842,)` |
| Test set | Kept untouched; no test images loaded or modified |

Normalization equation used:

```text
x_normalized = x / 255.0
```

where `x` is the original pixel value in `[0, 255]`.

## 8. Image Augmentation

Augmentation was configured in:

```text
notebooks/03_augmentation.ipynb
```

and reused identically during training in `04_cnn_training.ipynb`.

Actual augmentation, training-only:

| Parameter | Value |
|---|---|
| `rotation_range` | 10 degrees |
| `width_shift_range` | 0.08, ±8% |
| `height_shift_range` | 0.08, ±8% |
| `zoom_range` | 0.10, ±10% |
| `horizontal_flip` | `False`, disabled |
| `fill_mode` | `nearest` |
| Generator batch size | 32 |
| Shuffle | `True`, seed 42 |
| Validation augmentation | Disabled |
| Test augmentation | Disabled |

Why augmentation is useful here: traffic-sign photos differ in camera angle, position, distance, and cropping. Small rotations, shifts, and zooms create realistic variations of the same sign, helping the CNN rely less on exact position or size. Horizontal flipping is disabled because mirroring can change the meaning of directional signs, for example left versus right arrows.

The augmentation notebook only visualizes and checks augmented batches. It does not train the CNN.

## 9. CNN Model Architecture

The model is defined in:

```text
notebooks/04_cnn_training.ipynb
```

and saved as:

```text
models/traffic_sign_cnn.keras
```

### 9.1 Architecture

| Order | Layer | Configuration | Output shape | Parameters |
|---|---|---|---:|---:|
| Input | `Input` | shape `(32, 32, 3)` | `(32, 32, 3)` | 0 |
| 1 | `Conv2D` | 32 filters, `(3, 3)`, ReLU, `padding="same"` | `(32, 32, 32)` | 896 |
| 2 | `MaxPooling2D` | `(2, 2)` | `(16, 16, 32)` | 0 |
| 3 | `Conv2D` | 64 filters, `(3, 3)`, ReLU, `padding="same"` | `(16, 16, 64)` | 18,496 |
| 4 | `MaxPooling2D` | `(2, 2)` | `(8, 8, 64)` | 0 |
| 5 | `Conv2D` | 128 filters, `(3, 3)`, ReLU, `padding="same"` | `(8, 8, 128)` | 73,856 |
| 6 | `MaxPooling2D` | `(2, 2)` | `(4, 4, 128)` | 0 |
| 7 | `Flatten` | — | `(2048,)` | 0 |
| 8 | `Dense` | 128 units, ReLU | `(128,)` | 262,272 |
| 9 | `Dropout` | rate 0.50 | `(128,)` | 0 |
| 10 | `Dense` | 43 units, Softmax | `(43,)` | 5,547 |

The output uses Softmax:

```text
softmax(z)_i = exp(z_i) / sum_j exp(z_j)
```

### 9.2 Model statistics

There are two relevant parameter counts because the saved `.keras` file includes optimizer state:

| Statistic | Verified value |
|---|---|
| Trainable parameters, architecture only | 361,067, reported as 1.38 MB in a fresh summary |
| Total parameters after loading saved model | 1,083,203, reported as 4.13 MB, including 722,136 optimizer parameters |
| Trainable parameters after loading | 361,067 |
| Non-trainable parameters | 0 |
| Saved model file size | 4,380,536 bytes, about 4.18 MiB |

The requested “1,083,203 total parameters / approximately 4.13 MB” therefore matches the loaded saved model summary, not the CNN architecture alone.

## 10. Model Training

Training was performed in:

```text
notebooks/04_cnn_training.ipynb
```

| Item | Verified value |
|---|---|
| Training data | `X_train (31367, 32, 32, 3)`, `y_train (31367,)` |
| Validation data | `X_val (7842, 32, 32, 3)`, `y_val (7842,)`, no augmentation |
| Augmentation | Training generator only, same settings as Section 8 |
| Optimizer | `adam` |
| Loss | `sparse_categorical_crossentropy` |
| Metrics | `accuracy` |
| Epochs | 10 |
| Batch size | 32 |
| Callbacks | None used; no early stopping, checkpoint callback, or learning-rate schedule in the code |
| Validation handling | Passed as `(X_val, y_val)` to `model.fit` |
| Model saving | `model.save("models/traffic_sign_cnn.keras")` |

Optimizer details beyond the string `"adam"`, such as learning rate: **Not available** in the notebook.

Final epoch log values:

| Epoch | Train acc | Train loss | Val acc | Val loss |
|---:|---:|---:|---:|---:|
| 1 | 0.3355 | 2.2933 | 0.6811 | 0.9954 |
| 2 | 0.6627 | 1.0086 | 0.8942 | 0.3256 |
| 3 | 0.7967 | 0.5999 | 0.9491 | 0.1728 |
| 4 | 0.8600 | 0.4152 | 0.9700 | 0.1020 |
| 5 | 0.8929 | 0.3204 | 0.9841 | 0.0605 |
| 6 | 0.9202 | 0.2439 | 0.9818 | 0.0606 |
| 7 | 0.9314 | 0.2124 | 0.9852 | 0.0427 |
| 8 | 0.9416 | 0.1786 | 0.9907 | 0.0295 |
| 9 | 0.9506 | 0.1498 | 0.9899 | 0.0338 |
| 10 | 0.9572 | 0.1322 | 0.9895 | 0.0330 |

## 11. Training Results

| Metric | Value |
|---|---|
| Final training accuracy | 95.72% |
| Final validation accuracy | 98.95% |
| Final training loss | 0.1322 |
| Final validation loss | 0.0330 |

The accuracy curves rise steadily and the loss curves fall steadily over 10 epochs. Validation accuracy is above training accuracy in the later epochs. This is commonly observed when dropout and augmentation make training artificially harder: dropout rate 0.50 is active during training but not during validation, and augmented distorted images are used for training while clean images are used for validation.

The small final gap between validation loss 0.0330 and the stable high validation accuracy indicates no severe overfitting within these 10 epochs. Longer training behavior was not tested in the notebook.

## 12. Test Evaluation

Test evaluation was performed in:

```text
notebooks/05_test_evaluation.ipynb
```

The test set was completely untouched during training and tuning. It was loaded only for final evaluation with the same preprocessing as training: RGB, `32x32` LANCZOS resize, division by 255. No augmentation was applied.

| Metric | Value |
|---|---|
| Test samples | 12,630 |
| Test loss | 0.2129 |
| Test accuracy | 95.1465% |
| Macro precision | 92.8425% |
| Macro recall | 92.9763% |
| Macro F1-score | 92.4892% |
| Incorrect predictions | 613 |
| Correct predictions | 12,017, derived as 12,630 - 613 |

Accuracy:

```text
accuracy = correct predictions / total predictions
```

Macro metrics are unweighted means over the 43 classes. They treat small and large classes equally, which is useful because the dataset is imbalanced.

Also verified:

- `sklearn.metrics.classification_report` generated with 4 decimals.
- `sklearn.metrics.confusion_matrix` generated for all 43 classes.
- Confusion-matrix figure saved to `outputs/confusion_matrix.png`, verified present at 150,247 bytes.
- 12 incorrect examples displayed in the notebook; full error count is 613.

## 13. Class-wise Performance

The full 43-class report is generated in `05_test_evaluation.ipynb`. Performance varies between classes. Overall macro averages are about 92.84% precision, 92.98% recall, and 92.49% F1, below the 95.15% overall accuracy because smaller classes count equally in macro averaging.

Notable verified results, with support counts:

| Class | Precision | Recall | F1-score | Support | Note |
|---:|---:|---:|---:|---:|---|
| 16 | 1.0000 | 1.0000 | 1.0000 | 150 | Perfect on test split |
| 9 | 0.9938 | 0.9979 | 0.9958 | 480 | Very high |
| 14 | 1.0000 | 0.9889 | 0.9944 | 270 | Very high |
| 10 | 0.9969 | 0.9894 | 0.9932 | 660 | Very high |
| 13 | 0.9917 | 0.9931 | 0.9924 | 720 | Very high |
| 27 | 0.7895 | 0.5000 | 0.6122 | 60 | Lowest F1; half of its 60 test images missed |
| 30 | 0.6296 | 0.6800 | 0.6538 | 150 | Low precision and recall |
| 21 | 0.8108 | 0.6667 | 0.7317 | 90 | Low recall |
| 20 | 0.6338 | 1.0000 | 0.7759 | 90 | All found, but many false positives |
| 39 | 1.0000 | 0.6667 | 0.8000 | 90 | Low recall |
| 18 | 0.9939 | 0.8333 | 0.9066 | 390 | High precision, lower recall |

Some difficult or underrepresented classes have clearly lower F1 scores. Per-class values for all other classes are available in the notebook output and should be copied from there if a complete table is needed for the report.

## 14. Prediction Demo

Demo notebook:

```text
notebooks/06_image_prediction.ipynb
```

The user uploads an image directly in the notebook through an `ipywidgets.FileUpload` widget accepting `.png`, `.jpg`, and `.jpeg`. The notebook:

1. Loads `models/traffic_sign_cnn.keras`.
2. Converts the upload to RGB.
3. Resizes it to `32x32` with LANCZOS.
4. Normalizes with `/ 255.0`.
5. Adds a batch dimension and calls `model.predict`.
6. Outputs predicted class ID with `argmax`.
7. Outputs confidence with `max` probability.
8. Displays the original uploaded image.

Important: the original dataset does **not** provide human-readable class names. The prediction notebook defines its own 43-entry `class_names` dictionary for display only and checks it against representative training images. That mapping is not part of `Train.csv`, `Test.csv`, or `Meta.csv`.

## 15. Project Workflow

```text
Dataset
↓
Dataset Exploration
↓
Train/Validation Split
↓
Image Resizing + Normalization
↓
Image Augmentation
↓
Custom CNN Training
↓
Validation
↓
Saved CNN Model
↓
Test Evaluation
↓
Uploaded Image Prediction
```

## 16. Technologies Used

Verified from notebook imports and `requirements.txt`:

| Technology | Use |
|---|---|
| Python | Project language; version not pinned, **Not available** |
| TensorFlow/Keras | CNN definition, training, model saving/loading, `ImageDataGenerator`; version unpinned, **Not available** |
| NumPy | Arrays, labels, prediction processing |
| Pandas | Reading `Train.csv`, `Test.csv`, `Meta.csv` |
| Pillow, PIL | Image loading, RGB conversion, resizing, prediction preprocessing |
| Matplotlib | Accuracy/loss plots, sample grids, confusion-matrix and error plots |
| Seaborn | Plot styling, histograms, confusion-matrix heatmap |
| Scikit-learn | Stratified split, precision/recall/F1, classification report, confusion matrix |
| Jupyter Notebook | All six notebooks |
| ipywidgets | Upload widget in `06_image_prediction.ipynb` |

`requirements.txt` also lists `opencv-python`, but no `cv2` import was found in notebook cell sources. No other backend or deployment framework is used.

## 17. Project Structure

Actual repository structure:

```text
traffic-sign-recogonition/
├── dataset/
│   ├── Train/              # 43 folders 0-42, 39,209 PNGs (local only, gitignored)
│   ├── Test/               # 12,630 PNGs + empty GT-final_test.csv (local only, gitignored)
│   ├── Meta/               # 43 class images + lock files (local only, gitignored)
│   ├── Train.csv
│   ├── Test.csv
│   └── Meta.csv
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_data_preparation.ipynb
│   ├── 03_augmentation.ipynb
│   ├── 04_cnn_training.ipynb
│   ├── 05_test_evaluation.ipynb
│   └── 06_image_prediction.ipynb
├── models/
│   └── traffic_sign_cnn.keras
├── outputs/
│   └── confusion_matrix.png
├── README.md
├── requirements.txt
└── .gitignore
```

Notes:

- There is no `src/` directory in the current repository.
- `dataset/` is listed in `.gitignore` but present locally for this project.
- Only the files above were verified. Do not assume additional scripts, configs, or live-inference apps exist.

## 18. How to Run

1. Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

2. Install requirements:

```bash
pip install -r requirements.txt
```

3. Start Jupyter and run the notebooks in order:

```bash
jupyter notebook
```

Run:

```text
01_dataset_exploration.ipynb
02_data_preparation.ipynb
03_augmentation.ipynb
04_cnn_training.ipynb
05_test_evaluation.ipynb
```

4. For single-image prediction, open and run:

```text
notebooks/06_image_prediction.ipynb
```

Use the upload widget to choose a traffic-sign image. The notebook displays the image and prints predicted class and confidence. No command-line arguments or file paths are required.

## 19. Limitations

Only real limitations observed in the project:

- The dataset does not contain human-readable class names; only numeric class IDs plus `ShapeId`, `ColorId`, and `SignId` are provided.
- The model is trained on cropped traffic-sign images, not full road scenes; detection/localization is not handled.
- The current prediction demo uses uploaded images, not live webcam or video.
- Class performance is not uniform; classes such as 27 and 30 have substantially lower F1 scores than the average.
- ROI columns in the CSVs are unused; full images are resized directly.
- Training was run for a fixed 10 epochs with no callbacks or systematic hyperparameter search documented.

## 20. Future Improvements

Realistic next steps, none already implemented:

- Real-time webcam/video inference.
- Better handling of difficult and underrepresented classes, for example class weighting, balanced sampling, or targeted data collection.
- Improved augmentation that preserves sign semantics, possibly including brightness/contrast variation relevant to road scenes.
- Verification of the display-only class-name mapping against an authoritative sign list before use in reports or user-facing output.
- Object detection and localization so full road images can be processed, not only cropped signs.

## 21. Conclusion

A custom CNN with three convolutional blocks, one 128-unit dense layer, dropout 0.50, and a 43-way Softmax output was trained on 31,367 augmented traffic-sign images and validated on 7,842 images. Final training accuracy was 95.72% and validation accuracy was 98.95%. On a completely separate test set of 12,630 images, the saved model achieved 95.1465% accuracy with 0.2129 loss and 92.4892% macro F1-score, with 613 incorrect predictions. Performance is strong on average but varies by class, and the saved model plus upload notebook provide a simple reproducible baseline for cropped traffic-sign classification.
