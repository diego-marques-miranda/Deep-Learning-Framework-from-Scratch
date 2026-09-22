import matplotlib.pyplot as plt
import numpy as np


def _validate_history(history, metric_names):
	if "loss" not in history or "val_loss" not in history:
		raise ValueError("history must contain 'loss' and 'val_loss'.")

	if len(history["loss"]) != len(history["val_loss"]):
		raise ValueError("'loss' and 'val_loss' must have the same length.")

	for metric_name in metric_names:
		if metric_name not in history:
			raise ValueError(f"Metric '{metric_name}' was not found in history.")
		validation_name = f"val_{metric_name}"
		if validation_name not in history:
			raise ValueError(
				f"Metric '{metric_name}' is missing '{validation_name}'."
			)
		if len(history[metric_name]) != len(history[validation_name]):
			raise ValueError(
				f"'{metric_name}' and '{validation_name}' must have the same length."
			)


def _add_hover(figure, axes_lines, epochs):
	annotations = []

	for axis, lines in axes_lines:
		annotation = axis.annotate(
			"",
			xy=(0, 0),
			xytext=(12, 12),
			textcoords="offset points",
			bbox={"boxstyle": "round", "fc": "white", "alpha": 0.9},
			arrowprops={"arrowstyle": "->"}
		)
		annotation.set_visible(False)
		annotations.append(annotation)

	def on_move(event):
		if event.inaxes is None:
			for annotation in annotations:
				annotation.set_visible(False)
			figure.canvas.draw_idle()
			return

		annotation_index = next(
			(index for index, (axis, _) in enumerate(axes_lines)
			 if axis is event.inaxes),
			None
		)
		if annotation_index is None:
			return

		annotation = annotations[annotation_index]
		selected = None
		for line in axes_lines[annotation_index][1]:
			contains, details = line.contains(event)
			indices = details.get("ind")
			if contains and indices is not None and len(indices) > 0:
				selected = (line, indices[0])
				break

		if selected is None:
			annotation.set_visible(False)
		else:
			line, point_index = selected
			x_value = epochs[point_index]
			y_value = line.get_ydata()[point_index]
			annotation.xy = (x_value, y_value)
			annotation.set_text(
				f"Epoch: {x_value}\n{line.get_label()}: {y_value:.6f}"
			)
			annotation.set_visible(True)

		figure.canvas.draw_idle()

	figure.canvas.mpl_connect("motion_notify_event", on_move)


def _plot_panel(axis, epochs, title, train_values, validation_values):
	train_line, = axis.plot(epochs, train_values, label="Train")
	validation_line, = axis.plot(
		epochs, validation_values, label="Validation"
	)

	axis.text(
		0.01,
		0.97,
		f"Train: {train_values[-1]:.6f}\nValidation: {validation_values[-1]:.6f}",
		transform=axis.transAxes,
		ha="left",
		va="top",
		fontsize="small",
		bbox={"boxstyle": "round", "fc": "white", "alpha": 0.75}
	)
	axis.set_title(title)
	axis.set_ylabel(title)
	axis.legend()
	axis.grid(True)
	return train_line, validation_line


def plot_history(history, metrics=None, save_path=None):
	"""Plot training and validation loss and metrics over the epochs."""
	all_metric_names = [
		name for name in history
		if name not in {"loss", "val_loss"} and not name.startswith("val_")
	]
	metric_names = all_metric_names if metrics is None else list(metrics)
	_validate_history(history, metric_names)

	epochs = range(1, len(history["loss"]) + 1)
	panel_count = len(metric_names) + 1
	figure, axes = plt.subplots(
		panel_count,
		1,
		figsize=(10, min(36, max(5, 3.5 * panel_count))),
		sharex=True,
		squeeze=False
	)
	axes = axes.ravel()
	axes_lines = []

	axes_lines.append((
		axes[0],
		_plot_panel(axes[0], epochs, "Loss", history["loss"], history["val_loss"])
	))
	for axis, metric_name in zip(axes[1:], metric_names):
		axes_lines.append((
			axis,
			_plot_panel(
				axis,
				epochs,
				metric_name,
				history[metric_name],
				history[f"val_{metric_name}"]
			)
		))

	axes[-1].set_xlabel("Epoch")
	_add_hover(figure, axes_lines, epochs)
	figure.tight_layout()
	if save_path is not None:
		figure.savefig(save_path, bbox_inches="tight")
	plt.show()
	return figure

class ConfusionMatrix:
    """Accumulates a multiclass confusion matrix across batches."""

    def __init__(self):
        self.reset()

    def update(self, y_true, y_pred):
        """
        Accumulates confusion matrix counts for a batch.

        ``y_true`` contains integer class labels.
        ``y_pred`` is expected to contain class probabilities or scores.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        if y_pred.ndim != 2:
            raise ValueError(
                "y_pred must be a 2D array with shape "
                "(n_samples, n_classes)."
            )

        num_classes = y_pred.shape[1]
        predicted_classes = np.argmax(y_pred, axis=1)

        if self.num_classes is None:
            self.num_classes = num_classes
            self.matrix = np.zeros(
                (num_classes, num_classes),
                dtype=np.int64
            )

        if num_classes != self.num_classes:
            raise ValueError(
                "Number of classes in y_pred changed between updates."
            )

        if len(y_true) != len(predicted_classes):
            raise ValueError(
                "y_true and y_pred must contain the same number of samples."
            )

        if np.any(y_true < 0) or np.any(y_true >= self.num_classes):
            raise ValueError(
                "y_true contains class labels outside the valid range."
            )

        np.add.at(
            self.matrix,
            (y_true, predicted_classes),
            1
        )

    def result(self):
        """Returns the accumulated confusion matrix."""
        if self.matrix is None:
            return np.zeros((0, 0), dtype=np.int64)

        return self.matrix.copy()

    def reset(self):
        """Clears the accumulated confusion matrix."""
        self.num_classes = None
        self.matrix = None