# 🌿 Challenge 1 - Plant Classification

Classification problem on a plant dataset.

This is an image classification project where a Convolutional Neural Network (CNN) predicts the plant class from an input image.

---

## ⚡ TL;DR

- **Task:** Multi-class image classification (plants)
- **Input:** RGB images organized in subfolders (one folder per class)
- **Output:** Class label
- **Metric:** Top-1 accuracy (optionally per-class F1/recall)

---

## 🗂 Repository Structure

```
📁 data/                     # Dataset folder
📁 src/                      # Python modules (.py)
📁 notebooks/                # Jupyter notebooks (.ipynb)
📄 README.md
```

---

## 🧬 Data Layout

```
data/
└── FullPlantDataset/
    ├── training_sample/         # Default sample used by code
    ├── training.zip             # Used to setup the FULL dataset manually
    └── training/                # Unzipped full dataset goes here
        └── PutFoldersHere.txt   # Placeholder file: put plant folders here
```

> 📌 Inside `data/FullPlantDataset/training/`, the file `PutFoldersHere.txt` is only a placeholder.

---

## 📥 Download Full Dataset

> ⚠️ Due to GitHub's file size limit (100MB), the full dataset is not stored here.

You can download it from Google Drive:

👉 **[Download training.zip (Google Drive)](https://drive.google.com/file/d/1XGAmzYxqwmaSk5lDfro7SUBy2CwbW7lr/view?usp=sharing)**

Once downloaded, place it in the following path:

```
data/FullPlantDataset/training.zip
```

Then unzip it manually. After extraction, the folder structure should be:

```
data/FullPlantDataset/
└── training/
    ├── Apple/
    ├── Blueberry/
    └── ...
```

---

## 🙋‍♂️ Author

Giacomo Da Re – [GitHub Profile](https://github.com/Giacomo98DaRe)
