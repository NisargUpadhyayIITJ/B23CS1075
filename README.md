# MLOps Minor Exam — B23CS1075

**Nisarg Upadhyay | B23CS1075**

Image classification pipeline using **ResNet-18** on a 10-class dataset, with training, evaluation, and Docker support.

---

## Project Structure

```
B23CS1075/
├── train.py              # Train ResNet-18 and save model weights
├── evaluate.py           # Evaluate model with metrics & confusion matrix
├── Dockerfile            # Docker image for running evaluation
├── requirements.txt      # Python dependencies
├── trained_model.pth     # Saved model weights (generated after training)
├── training_curves.png   # Loss & accuracy plots (generated after training)
├── train_confusion_matrix.png  # Training confusion matrix (generated)
├── confusion_matrix.png  # Test confusion matrix (generated after evaluation)
└── data/
    ├── train/            # Training images (ImageFolder format)
    └── test/             # Test images (ImageFolder format)
```

---

## Setup

### Prerequisites

- Python 3.10+
- pip

### Install Dependencies

```bash
cd B23CS1075
pip install -r requirements.txt
```

---

## Usage

### 1. Train the Model

```bash
python train.py
```

- Trains ResNet-18 for 3 epochs on `data/train/`
- Saves model weights to `trained_model.pth`
- Generates `training_curves.png` (loss & accuracy over epochs)
- Generates `train_confusion_matrix.png`

### 2. Evaluate the Model

```bash
python evaluate.py
```

- Loads trained weights from `trained_model.pth`
- Reports overall accuracy, F1 score, and classification report
- Prints class-wise accuracy
- Saves `confusion_matrix.png`
- Runs a random single-image prediction with confidence score

---

## Docker

### Using the Dockerfile

```bash
cd B23CS1075
docker build -t evaluate-model .
docker run evaluate-model
```

### Without a Dockerfile (manual method)

```bash
# Start an interactive container
docker run -it --name eval-container python:3.10-slim /bin/bash

# Inside the container, install dependencies
pip install torch torchvision numpy scikit-learn Pillow matplotlib seaborn

# In a second terminal, copy files into the container
docker cp evaluate.py eval-container:/app/evaluate.py
docker cp trained_model.pth eval-container:/app/trained_model.pth
docker cp data/test eval-container:/app/data/test

# Back in the container
cd /app && exit

# Commit and run
docker commit eval-container evaluate-image
docker run --workdir /app evaluate-image python evaluate.py
```

---

## Results

| Metric           | Description                          |
|------------------|--------------------------------------|
| Overall Accuracy | Percentage of correct predictions    |
| F1 Score (Macro) | Macro-averaged F1 across all classes |
| Class-wise Accuracy | Per-class accuracy breakdown      |
| Confusion Matrix | Visual heatmap of predictions        |

---

## Tech Stack

- **PyTorch** — model training & inference
- **torchvision** — ResNet-18, transforms, ImageFolder
- **scikit-learn** — accuracy, F1, confusion matrix, classification report
- **matplotlib + seaborn** — training curves & confusion matrix plots
