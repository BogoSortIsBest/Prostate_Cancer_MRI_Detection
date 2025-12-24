# Prostate Cancer Detection from MRI using 3D CNN

This project explores the use of a 3D Convolutional Neural Network (CNN) to predict clinically significant prostate cancer (csPCa) from volumetric prostate MRI scans.

Using a dataset of `.mha` MRI volumes and corresponding clinical labels, the model achieved approximately **70% classification accuracy** on a held-out test set.

---

## Overview

- Input: 3D prostate MRI scans (`.mha` format) - I utilized ~ 2000 public classified images
- Task: Binary / multi-class classification of prostate cancer presence or severity
- Model: 3D Convolutional Neural Network built with TensorFlow/Keras
- Labels: Derived either from folder names or from an Excel metadata file (depending on pipeline)
- Output: Class probabilities + accuracy metrics

---

## Files

| File | Description |
|------|------------|
| `main.py` | Loads MRI volumes, preprocesses them, trains and evaluates a 3D CNN |
| `cnn.py` | Alternate pipeline that uses an Excel sheet to map patient IDs to labels |
| `best_model.h5` | Saved model weights from best validation performance |
| *(not included)* | MRI volumes (`.mha`) and label spreadsheet |

---

## Dataset Format

### MRI files

- Format: `.mha` volumetric medical images

Expected structure (folder-based labels):

```
ml_train_images/
  ├── 0/
  ├── 1/
ml_test_images/
  ├── ...
```

OR (Excel-based labels):

```
1234_scan.mha
5678_scan.mha
```

Where `1234` corresponds to a `patient_id` in the Excel label sheet.

---

### Labels (Excel pipeline)

Excel file must contain:

| Column | Meaning |
|--------|----------|
| patient_id | Numeric ID corresponding to file names |
| case_csPCa | Cancer label |

---

## Model Architecture

- Conv3D (32 filters, 3×3×3 kernel)
- MaxPooling3D
- Conv3D (64 filters)
- MaxPooling3D
- Flatten
- Dense (128 units, ReLU)
- Dense (Softmax output)

Loss: Categorical Crossentropy  
Optimizer: Adam  
Metric: Accuracy

---

## Preprocessing

- MRI volumes converted to NumPy arrays
- Pixel values normalized to [0,1]
- Labels encoded and converted to one-hot vectors
- Train/validation split: 80/20

---

## How to Run

### 1. Install dependencies

```
pip install numpy pandas scikit-learn SimpleITK tensorflow openpyxl
```

### 2. Update file paths

Inside `main.py` or `cnn.py`:

```
train_folder = r'E:\ml_train_images'
test_folder = r'E:\ml_test_images'
excel_path = r'path\to\marksheet.xlsx'
```

### 3. Run training

```
python main.py
```

or

```
python cnn.py
```

---

## Results

| Metric | Value |
|--------|--------|
| Test Accuracy | ~70% |
| Epochs | 10 |
| Batch Size | 32 |

---

## Limitations

- Small dataset ~ 2000 Public MRI data
- No cross-validation
- No MRI standardization
- No segmentation or ROI extraction
- No data augmentation

---

## Future Improvements

- MRI normalization
- Prostate segmentation
- Data augmentation
- Deeper architectures (ResNet3D, attention models)
- ROC/AUC and confusion matrix evaluation

---

## Author

Built by Ishaan Singh
