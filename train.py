from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import arff

import activation_functions
from data.dataloader import DataLoader
from data.utils import train_test_split
from data.scaler import MinMaxScaler
import layers
import loss_functions
from metrics.metrics import (
    Accuracy,
    MacroPrecision,
    MacroRecall,
    MacroF1
)
from metrics.visualization import ConfusionMatrix
from metrics.visualization import plot_history
from model import Model
import optimizers


# ============================================================
# Experiment configuration
# ============================================================

experiment_id = "E16"
experiment_group = "optimizer_selection"

seed = 42

architecture = [27, 32, 16, 7]
optimizer_name = "Adam"
learning_rate = 0.01
batch_size = 32

max_epochs = 500

early_stopping = True
monitor = "val_loss"
mode = "min"
patience = 30
min_delta = 0.0
restore_best_weights = True


# ============================================================
# Random generators
# ============================================================

split_rng = np.random.default_rng(seed)
train_loader_rng = np.random.default_rng(seed + 1)
val_loader_rng = np.random.default_rng(seed + 2)
model_rng = np.random.default_rng(seed + 3)


# ============================================================
# Dataset
# ============================================================

file_path = Path(__file__).parent / "steel_plates_faults.arff"

data, meta = arff.loadarff(file_path)
df = pd.DataFrame(data)

class_mapping = {
    0: "Pastry",
    1: "Z_Scratch",
    2: "K_Scatch",
    3: "Stains",
    4: "Dirtiness",
    5: "Bumps",
    6: "Other_Faults"
}

indicator_columns = [
    "V28",
    "V29",
    "V30",
    "V31",
    "V32",
    "V33"
]

y = pd.Series(6, index=df.index)

for class_index, column in enumerate(indicator_columns):
    y.loc[df[column] == 1] = class_index

feature_columns = [f"V{i}" for i in range(1, 28)]

X = df[feature_columns].to_numpy()
y = y.to_numpy()


# ============================================================
# Train / Validation split
# ============================================================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    rng=split_rng,
    stratify=y
)


# ============================================================
# Preprocessing
# ============================================================

scaler = MinMaxScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)


# ============================================================
# DataLoaders
# ============================================================

train_loader = DataLoader(
    X_train,
    y_train,
    batch_size=batch_size,
    rng=train_loader_rng
)

val_loader = DataLoader(
    X_val,
    y_val,
    batch_size=batch_size,
    rng=val_loader_rng
)


# ============================================================
# Model
# ============================================================

model = Model()

for layer_index, (input_size, output_size) in enumerate(
    zip(architecture[:-1], architecture[1:])
):

    is_output_layer = layer_index == len(architecture) - 2

    activation = (
        activation_functions.Softmax()
        if is_output_layer
        else activation_functions.ReLU()
    )

    model.add(
        layers.Layer_Dense(
            input_size,
            output_size,
            activation_func=activation,
            rng=model_rng
        )
    )


# ============================================================
# Model parameters
# ============================================================

total_parameters = 0

for layer in model.layers:
    if hasattr(layer, "weights"):
        total_parameters += layer.weights.size
        total_parameters += layer.biases.size


# ============================================================
# Optimizer
# ============================================================

optimizer_classes = {
    "SGD": optimizers.SGD,
    "Momentum": optimizers.SGD_Momentum,
    "RMSprop": optimizers.RMSprop,
    "Adam": optimizers.Adam
}

if optimizer_name not in optimizer_classes:
    raise ValueError(
        f"Unknown optimizer: {optimizer_name}. "
        f"Available options: {list(optimizer_classes)}"
    )

optimizer_class = optimizer_classes[optimizer_name]

optimizer = optimizer_class(
    learning_rate=learning_rate
)


# ============================================================
# Compile
# ============================================================

model.compile(
    loss=loss_functions.CrossEntropy(),
    optimizer=optimizer
)


# ============================================================
# Metrics
# ============================================================

train_metrics = [
    Accuracy(),
    MacroPrecision(),
    MacroRecall(),
    MacroF1()
]

val_metrics = [
    Accuracy(),
    MacroPrecision(),
    MacroRecall(),
    MacroF1()
]


# ============================================================
# Training
# ============================================================

model.fit(
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    epochs=max_epochs,
    train_metrics=train_metrics,
    val_metrics=val_metrics,
    early_stopping=early_stopping,
    monitor=monitor,
    mode=mode,
    patience=patience,
    min_delta=min_delta,
    restore_best_weights=restore_best_weights
)


# ============================================================
# History
# ============================================================

history = model.history

metric_names = [
    name
    for name in history
    if name not in {"loss", "val_loss"}
    and not name.startswith("val_")
]


# ============================================================
# Last recorded epoch
# ============================================================

last_train_loss = history["loss"][-1]
last_val_loss = history["val_loss"][-1]


# ============================================================
# Selected checkpoint
# ============================================================

selected_train_values = {}
selected_val_values = {}

if model.best_epoch is not None:

    best_epoch_index = model.best_epoch - 1

    for metric_name in metric_names:
        selected_train_values[metric_name] = (
            history[metric_name][best_epoch_index]
        )

        selected_val_values[metric_name] = (
            history[f"val_{metric_name}"][best_epoch_index]
        )


# ============================================================
# Generalization gaps
# ============================================================

last_gaps = {
    "Loss": last_val_loss - last_train_loss
}

for metric_name in metric_names:
    train_value = history[metric_name][-1]
    val_value = history[f"val_{metric_name}"][-1]

    last_gaps[metric_name] = train_value - val_value


selected_gaps = {}

if model.best_epoch is not None:

    selected_gaps["Loss"] = (
        selected_val_values["loss"]
        if False
        else selected_val_values.get("Loss")
    )

    selected_gaps = {
        "Loss": (
            selected_val_values
            if False
            else history["val_loss"][best_epoch_index]
            - history["loss"][best_epoch_index]
        )
    }

    for metric_name in metric_names:
        selected_gaps[metric_name] = (
            selected_train_values[metric_name]
            - selected_val_values[metric_name]
        )


# ============================================================
# Best validation results
# ============================================================

best_val_loss = min(history["val_loss"])
best_val_loss_epoch = history["val_loss"].index(
    best_val_loss
) + 1


# ============================================================
# Confusion Matrix
# ============================================================

confusion_matrix = ConfusionMatrix()

for X_batch, y_batch in val_loader:
    y_pred = model.predict(X_batch)
    confusion_matrix.update(y_batch, y_pred)

confusion_matrix_result = confusion_matrix.result()

class_names = list(class_mapping.values())

confusion_matrix_df = pd.DataFrame(
    confusion_matrix_result,
    index=class_names,
    columns=class_names
)


# ============================================================
# Experiment summary
# ============================================================

architecture_str = " → ".join(map(str, architecture))

print("\n" + "=" * 60)
print("EXPERIMENT SUMMARY")
print("=" * 60)

print(f"Experiment ID: {experiment_id}")
print(f"Experiment Group: {experiment_group}")

print("\nConfiguration:")
print(f"  Architecture: {architecture_str}")
print(f"  Parameters: {total_parameters}")
print(f"  Optimizer: {optimizer_name}")
print(f"  Learning Rate: {learning_rate}")
print(f"  Batch Size: {batch_size}")
print(f"  Max Epochs: {max_epochs}")
print(f"  Seed: {seed}")


print("\nEarly Stopping:")
print(f"  Enabled: {early_stopping}")

if early_stopping:
    print(f"  Monitor: {monitor}")
    print(f"  Mode: {mode}")
    print(f"  Patience: {patience}")
    print(f"  Min Delta: {min_delta}")
    print(f"  Restore Best Weights: {restore_best_weights}")

print(f"  Epochs Trained: {model.epochs_trained}")
print(f"  Stopped Early: {model.stopped_early}")


print("\nSelected Checkpoint:")

if model.best_epoch is not None:
    print(f"  Monitor: {monitor}")
    print(f"  Best Value: {model.best_value:.4f}")
    print(f"  Best Epoch: {model.best_epoch}")
    print(f"  Best Weights Restored: {restore_best_weights}")

else:
    print("  No checkpoint selected.")


print("\nSelected Checkpoint Performance:")

if model.best_epoch is not None:

    selected_train_loss = history["loss"][best_epoch_index]
    selected_val_loss = history["val_loss"][best_epoch_index]

    print(f"  Train Loss: {selected_train_loss:.4f}")
    print(f"  Validation Loss: {selected_val_loss:.4f}")

    for metric_name in metric_names:

        print(
            f"  Train {metric_name}: "
            f"{selected_train_values[metric_name]:.4f}"
        )

        print(
            f"  Validation {metric_name}: "
            f"{selected_val_values[metric_name]:.4f}"
        )


print("\nBest Validation Loss:")
print(f"  Value: {best_val_loss:.4f}")
print(f"  Epoch: {best_val_loss_epoch}")


print("\nLast Recorded Epoch:")

print(f"  Train Loss: {last_train_loss:.4f}")
print(f"  Validation Loss: {last_val_loss:.4f}")

for metric_name in metric_names:

    print(
        f"  Train {metric_name}: "
        f"{history[metric_name][-1]:.4f}"
    )

    print(
        f"  Validation {metric_name}: "
        f"{history[f'val_{metric_name}'][-1]:.4f}"
    )


print("\nGeneralization Gap at Selected Checkpoint:")

selected_gaps = {}

if model.best_epoch is not None:

    selected_gaps = {
        "Loss": (
            history["val_loss"][best_epoch_index]
            - history["loss"][best_epoch_index]
        )
    }

    for metric_name in metric_names:
        selected_gaps[metric_name] = (
            selected_train_values[metric_name]
            - selected_val_values[metric_name]
        )


print("\nConfusion Matrix:")
print(confusion_matrix_df)

print("=" * 60)


# ============================================================
# Visualization
# ============================================================

plot_history(model.history)