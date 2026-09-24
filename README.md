# 🧠 Deep Learning Framework from Scratch

A lightweight, educational Deep Learning framework built from scratch using **Python and NumPy**.

No TensorFlow. No PyTorch. No Keras.

The goal of this project is to understand what happens **inside a Deep Learning framework** by implementing its fundamental components manually, from forward propagation and backpropagation to optimization, metrics and training utilities.

---

## 🎯 Project Goal

This project was created primarily as a **learning and experimentation environment**, not as a production-ready framework or a benchmark for model performance.

The main objective was to turn Deep Learning concepts that are usually hidden behind high-level libraries into explicit implementations that can be inspected, tested and experimented with.

The project focuses on understanding:

- Forward propagation
- Backpropagation and the chain rule
- Gradient descent
- Weight initialization
- Activation functions
- Loss functions
- Optimizers
- Training and validation workflows
- Early stopping
- Model evaluation
- Regression and multiclass classification

The experiments included in the repository are **test beds for validating the framework and studying how its components behave**, rather than attempts to build the best possible model for each dataset.

---

## ⚙️ Features

### Core Architecture

- Object-oriented `Model` class
- Dynamic layer stacking
- `add()`, `compile()`, `fit()` and `predict()` workflow
- Mini-batch training through `DataLoader`
- Training history tracking
- Early stopping
- Best-weight checkpoint restoration
- Model weight serialization with `.pkl`

### Layers

- Fully connected (`Layer_Dense`)

### Activation Functions

- ReLU
- Leaky ReLU
- Sigmoid
- Tanh
- GELU
- Linear
- Softmax

### Loss Functions

- Mean Squared Error (MSE)
- Cross-Entropy

### Optimizers

- SGD
- Momentum
- RMSprop
- Adam

### Metrics

Regression:

- MAPE
- R²

Classification:

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1
- Confusion Matrix

### Data & Training Utilities

- Train/test splitting
- Stratified splitting for classification
- Data loading and mini-batching
- Median imputation
- Feature scaling
- Reproducible random number generation

### Visualization

- Training and validation loss
- Training and validation metrics
- Interactive metric inspection

---

## 🧠 Framework Architecture

The framework follows a simple separation of responsibilities:

```text
Dataset
   │
   ▼
Data Preparation
   │
   ├── Train/Test Split
   ├── Imputation
   └── Scaling
   │
   ▼
DataLoader
   │
   ▼
Model
   │
   ├── Dense Layers
   ├── Activation Functions
   ├── Loss Function
   └── Optimizer
   │
   ▼
Training Loop
   │
   ├── Forward Pass
   ├── Loss Calculation
   ├── Backward Pass
   ├── Gradient Update
   └── Metrics
   │
   ▼
Training History
   │
   ├── Loss
   ├── Accuracy / Regression Metrics
   └── Validation Metrics
```

Preprocessing and data utilities are kept outside the `Model`, keeping the framework components separated and reusable.

---

## 🔬 Test Beds

The repository contains separate test beds designed to exercise different capabilities of the framework.

### Classification — Iris

```text
test_beds/classification/
```

Uses the Iris dataset to validate the classification workflow, including:

- Multiclass predictions
- Softmax probabilities
- Cross-Entropy loss
- Accuracy
- Backpropagation through the classification pipeline

Run with:

```bash
python -m test_beds.classification.classification
```

---

### Regression — California Housing

```text
test_beds/regression/
```

Uses the California Housing dataset to validate the regression workflow, including:

- Continuous target prediction
- MSE loss
- R²
- MAPE
- Feature preprocessing
- Mini-batch training

Run with:

```bash
python -m test_beds.regression.regression
```

---

### Multiclass Classification — Steel Plates Faults

```text
test_beds/multiclass/
```

This test bed uses the **Steel Plates Faults** dataset to exercise the framework on a more complex multiclass classification problem.

The model receives **27 descriptive features** and predicts one of **7 fault classes**:

```text
Pastry
Z_Scratch
K_Scatch
Stains
Dirtiness
Bumps
Other_Faults
```

The experiment was also used to validate classification-specific evaluation tools such as:

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1
- Confusion Matrix
- Early stopping
- Validation monitoring

Run with:

```bash
python -m test_beds.multiclass.multiclass
```

---

## 🧪 Experimental Approach

The experiments in this repository are intended to **validate the framework and study Deep Learning fundamentals**.

The goal was not to exhaustively optimize each dataset or produce state-of-the-art results.

Instead, controlled experiments were used to investigate questions such as:

- How does the learning rate affect convergence?
- How do different optimizers behave under the same architecture?
- How does network capacity affect training and validation performance?
- How does overfitting appear during training?
- How do different classification metrics describe model behavior?
- How does early stopping affect model selection?

For example, optimizer experiments were performed while keeping the architecture, learning rate, batch size and dataset split controlled.

This makes the experiments useful for understanding **how the framework behaves internally**, rather than simply obtaining the highest possible score.

---

## 📊 Model Evaluation

For classification experiments, accuracy is not used as the only evaluation metric.

The framework also implements:

```text
Precision
Recall
F1
```

using macro averaging across classes.

This makes it possible to inspect model behavior across different classes instead of relying only on the overall percentage of correct predictions.

The confusion matrix provides an additional view of the errors:

```text
                 Predicted
              ┌─────────────────────┐
              │                     │
      Actual  │  Class relationships │
              │     and errors       │
              │                     │
              └─────────────────────┘
```

For regression experiments, the framework includes:

- MSE for optimization
- R² for explained variance
- MAPE for relative prediction error

---

## 🛑 Early Stopping

The training loop supports early stopping based on a monitored validation metric.

Example configuration:

```python
model.fit(
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    epochs=500,
    early_stopping=True,
    monitor="val_loss",
    mode="min",
    patience=30,
    restore_best_weights=True
)
```

When the monitored validation metric stops improving, training can stop before reaching the maximum number of epochs.

The framework can also restore the weights from the best validation checkpoint.

This is particularly useful when studying the difference between:

```text
Training performance
        vs.
Validation performance
```

and observing overfitting during experiments.

---

## 🚀 Running the Project

### 1. Clone the repository

```bash
git clone https://github.com/diego-marques-miranda/Deep-Learning-Framework-from-Scratch.git
cd Deep-Learning-Framework-from-Scratch
```

### 2. Install dependencies

The project uses Python and NumPy for the core framework, with additional libraries for datasets, preprocessing and visualization.

```bash
pip install numpy pandas scipy matplotlib mplcursors
```

### 3. Run a test bed

From the project root:

```bash
python -m test_beds.classification.classification
```

```bash
python -m test_beds.regression.regression
```

```bash
python -m test_beds.multiclass.multiclass
```

Running the test beds as modules keeps the package structure consistent and allows the framework modules at the repository root to be imported correctly.

---

## 📁 Project Structure

```text
Deep-Learning-Framework-from-Scratch/
│
├── data/
│   ├── dataloader.py
│   ├── preprocessing.py
│   ├── scaler.py
│   └── utils.py
│
├── metrics/
│   ├── metrics.py
│   ├── evaluation.py
│   └── visualization.py
│
├── test_beds/
│   ├── classification/
│   │   ├── __init__.py
│   │   ├── classification.py
│   │   └── IRIS.csv
│   │
│   ├── regression/
│   │   ├── __init__.py
│   │   ├── regression.py
│   │   └── housing.csv
│   │
│   └── multiclass/
│       ├── __init__.py
│       ├── multiclass.py
│       └── steel_plates_faults.arff
│
├── tests/
│
├── activation_functions.py
├── layers.py
├── loss_functions.py
├── model.py
└── optimizers.py
```

---

## 🛠️ Tech Stack

| Category | Technology |
| --- | --- |
| Language | Python |
| Numerical Computing | NumPy |
| Data Processing | Pandas |
| Dataset Loading | SciPy |
| Visualization | Matplotlib |
| Interactive Plots | mplcursors |
| Deep Learning Framework | Built from scratch |
| Testing | Python test suite |

---

## 🔑 Key Design Decisions

### Framework components are implemented independently

Layers, activation functions, losses, optimizers, metrics and data utilities are separated into their own modules.

This makes it possible to combine different components without coupling the entire framework to a specific model or experiment.

---

### Preprocessing is kept outside the Model

Data preprocessing such as imputation, scaling and dataset splitting is handled independently from the neural network itself.

The model is responsible for training and inference, while preprocessing utilities handle data preparation.

This keeps the responsibilities of each component explicit.

---

### Classification targets use class indices

Multiclass targets are represented using integer class labels rather than requiring one-hot encoded targets.

The final layer produces class scores, Softmax converts them into probabilities, and the predicted class is obtained through `argmax` during evaluation.

---

### Validation is separated from training

The framework supports a dedicated validation dataloader during training.

This allows validation performance to be monitored independently of training performance and provides the basis for:

- Early stopping
- Best checkpoint selection
- Generalization analysis

---

### Metrics accumulate across batches

Metrics such as Macro Precision, Macro Recall and Macro F1 accumulate the required counts across batches before calculating the final metric.

This avoids incorrectly averaging individual batch metrics.

---

## ⚠️ Project Scope & Limitations

This project is intentionally educational.

It is not intended to compete with mature Deep Learning frameworks such as PyTorch or TensorFlow.

Current limitations include:

- The framework is focused primarily on educational experimentation.
- The layer abstraction currently centers on fully connected neural networks.
- There is no GPU acceleration.
- There is no automatic computational graph.
- The framework does not provide the ecosystem or optimization level of production Deep Learning libraries.
- The included datasets are test beds for framework validation, not production ML applications.
- Experimental results should not be interpreted as benchmarks against state-of-the-art models.

These limitations are intentional and keep the project focused on understanding the fundamental mechanics of Deep Learning.

---

## 🎓 What This Project Demonstrates

By building the framework and its test beds, the project covers the following workflow:

```text
Deep Learning Fundamentals
        ↓
Forward Propagation
        ↓
Loss Calculation
        ↓
Backpropagation
        ↓
Gradient Descent
        ↓
Optimizers
        ↓
Mini-Batch Training
        ↓
Validation
        ↓
Early Stopping
        ↓
Model Evaluation
        ↓
Regression & Classification
```

The project therefore serves as a practical implementation of the concepts that are normally hidden behind high-level Deep Learning APIs.

Instead of simply using a neural network library, the framework exposes the mechanics behind the training process.

---

## ✅ Project Status

This project is considered **complete** as an educational Deep Learning framework and experimentation environment.

The current implementation and test beds provide the functionality needed for the project's original objective:

> **Understand and implement the fundamental building blocks of Deep Learning from scratch.**

Further model optimization or additional architecture development is outside the current scope of the project.

---

## 👨‍💻 Author

**Diego Marques Miranda**

AI/ML & Data Science · Deep Learning · Data Engineering · Python

[GitHub](https://github.com/diego-marques-miranda)