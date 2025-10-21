# 🎓 Real-Time Egocentric Action Recognition using Gaze-Guided Attention

This repository contains the full implementation of my **Master’s Thesis** at *Sapienza University of Rome*, focused on **real-time egocentric action recognition** using RGB video streams enhanced with **human gaze information**.  
The model jointly learns from visual features and gaze patterns to classify fine-grained egocentric actions efficiently.

---

## 📁 Project Structure

```
├── install.sh                  # Automatic environment setup and data download
├── requirements.txt            # Python dependencies
├── Preprocessing.py            # Frame extraction and dataset verification
├── dataset.py                  # Custom PyTorch Dataset (IGazeDataset)
├── models.py                   # Feature Encoder + Transformer Classifier
├── train.py                    # Training pipeline and logging
├── train_split1_filtered.txt   # Train split (filtered EGTEA subset)
├── test_split1_filtered.txt    # Test split (filtered EGTEA subset)
└── Gaze_Data/                  # Gaze data (Frames will be downloaded via install.sh)
```

---

## ⚙️ Installation

Run the following command to automatically set up everything — including Miniconda, dependencies, and dataset download:

```bash
bash install.sh
```

This script will:
- Install **Miniconda** and create an environment `egtea-env`
- Install all dependencies from `requirements.txt`
- Download and process the **EGTEA Gaze+ dataset**
- Extract frames and generate ready-to-train splits

---

## 🧠 Model Overview

The architecture consists of two main components:

### 1. FeatureEncoder (ResNet-152 Backbone)
- Extracts spatial features from RGB frames  
- Predicts **attention maps** guided by gaze regions  
- Outputs verb and noun logits

### 2. TransformerClassifier
- Processes temporal frame sequences  
- Learns temporal dependencies across clips  
- Outputs final action predictions  

Each video clip contains **16 consecutive frames**, normalized and augmented with **Gaussian gaze heatmaps**.

---

## 🧩 Dataset Details

- **Dataset:** [EGTEA Gaze+](https://cbs.ic.gatech.edu/fpv/)
- **Modalities Used:** RGB + Gaze (ground truth during training)
- **Clip Length:** 16 frames
- **Labels:** Action, Verb, and Noun classes
- **Sampling:**
  - **Train:** Random sequence of 16 frames (clips)
  - **Test:** Up to 10 evenly spaced clips per video

---

## 🚀 Training

Once the dataset and environment are prepared:

```bash
python train.py
```

You can adjust hyperparameters such as:
- `num_epochs`
- `learning_rate`
- `batch_size`
- `etc.`
directly in `train.py`.

The model jointly trains:
- **Verb/Noun classification heads**
- **Gaze prediction branch**

---

## 📊 Results

- Trained using RGB + Gaze information  
- Converged in just **15 epochs**
- Achieved **state-of-the-art performance** among lightweight models

---

## 🧰 Requirements

Main dependencies:
```
torch
torchvision
numpy
opencv-python
tqdm
pandas
matplotlib
scipy
jupyter
```

All dependencies are listed in [`requirements.txt`](requirements.txt).

---

## 📬 Citation

If you use this repository, please cite:

> **Zhalgasbayev, Yerassyl.**  
> *Real-Time Egocentric Action Recognition using Gaze-Guided Attention.*  
> Master’s Thesis, Sapienza University of Rome, 2025.

---


## ⭐ Acknowledgements

This work builds upon the **EGTEA Gaze+ dataset** and previous studies in egocentric vision and attention modeling.  
Special thanks to **Prof. Luigi Cinque** and **Marco Raoul Marini** for their guidance and support.

---
