import os
import mlflow
import mlflow.keras

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, ConfusionMatrixDisplay,
    RocCurveDisplay, PrecisionRecallDisplay
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras import layers, Model, Input



def build_model(conv_base, config):
    inputs = keras.Input(shape=(224, 224, 3))

    # Data augmentation layer inside model
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ])

    x = data_augmentation(inputs)
    x = conv_base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(config["dense_units"], activation="relu")(x)
    x = layers.Dropout(config["dropout_rate"])(x)

    outputs = {
        "head": layers.Dense(1, activation="sigmoid", name="head")(x),
        "neck": layers.Dense(3, activation="softmax", name="neck")(x),
        "shoulder": layers.Dense(1, activation="sigmoid", name="shoulder")(x),
        "thoraric": layers.Dense(1, activation="sigmoid", name="thoraric")(x),
        "lumbar": layers.Dense(3, activation="softmax", name="lumbar")(x),
    }

    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=config.get("optimizer", "rmsprop"),
        loss=config["losses"],
        metrics=config["metrics"],
        loss_weights=config["loss_weights"],
    )

    return model


class CNNModelRunner:
    def __init__(self, model, val_ds, output_names, config=None):
        self.model = model
        self.val_ds = val_ds
        self.output_names = output_names
        self.config = config or {}

        self.metrics = {}
        self.y_val = {name: [] for name in output_names}
        self.y_prob = {name: [] for name in output_names}
        self.y_pred = {name: [] for name in output_names}
        self.y_val_np = {name: None for name in output_names}
        self.y_prob_np = {name: None for name in output_names}
        self.y_pred_np = {name: None for name in output_names}

    def evaluate(self):
        for name in self.output_names:
            self.y_val[name].clear()
            self.y_prob[name].clear()

        for x, y in self.val_ds:
            preds = self.model(x, training=False)
            for name in self.output_names:
                self.y_val[name].extend(y[name].numpy())
                self.y_prob[name].extend(preds[name].numpy())

        for name in self.output_names:
            self.y_val_np[name] = np.array(self.y_val[name])
            self.y_prob_np[name] = np.array(self.y_prob[name])
            if self.y_prob_np[name].shape[-1] == 1:
                self.y_pred_np[name] = (self.y_prob_np[name] > 0.5).astype(int)
                avg = "binary"
            else:
                self.y_pred_np[name] = np.argmax(self.y_prob_np[name], axis=1)
                avg = "weighted"

            self.metrics[name] = {
                "accuracy": accuracy_score(self.y_val_np[name], self.y_pred_np[name]),
                "precision": precision_score(self.y_val_np[name], self.y_pred_np[name], average=avg, zero_division=0),
                "recall": recall_score(self.y_val_np[name], self.y_pred_np[name], average=avg, zero_division=0),
                "f1": f1_score(self.y_val_np[name], self.y_pred_np[name], average=avg, zero_division=0),
                "log_loss": log_loss(self.y_val_np[name], self.y_prob_np[name]),
            }

        return self.metrics

    def log_to_mlflow(self, experiment_name=None):
        if experiment_name:
            mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name="cnn_multioutput"):
            # Log metrics
            for output, metric_dict in self.metrics.items():
                for metric_name, value in metric_dict.items():
                    mlflow.log_metric(f"{output}_{metric_name}", value)

            # Log the model
            mlflow.keras.log_model(self.model, artifact_path="model")

            # Log config as JSON artifact
            config_path = "config_logged.json"
            with open(config_path, "w") as f:
                json.dump(self.config, f, indent=2)
            mlflow.log_artifact(config_path)

    

    def plot_confusion_matrix(self, output_name, ax):
        ConfusionMatrixDisplay.from_predictions(self.y_val_np[output_name], self.y_pred_np[output_name], ax=ax, colorbar=False)
        ax.set_title(f"Confusion Matrix - {output_name}")

        

    def _plot_curve(self, display_fn, output_name, title, ax):
        y_val = self.y_val_np[output_name]
        y_prob = self.y_prob_np[output_name]

        if y_prob.shape[1] == 1:  # Binary classification
            display_fn.from_predictions(y_val, y_prob[:, 0], ax=ax)
        else:  # Multiclass classification
            n_classes = y_prob.shape[1]
            y_true_bin = label_binarize(y_val, classes=np.arange(n_classes))
            for i in range(n_classes):
                display_fn.from_predictions(
                    y_true_bin[:, i], y_prob[:, i], name=f"Class {i}", ax=ax
                )

        ax.set_title(f"{title} - {output_name}")

    def plot_roc_curve(self, output_name, ax):
        self._plot_curve(RocCurveDisplay, output_name, "ROC Curve", ax)

    def plot_precision_recall(self, output_name, ax):
        self._plot_curve(PrecisionRecallDisplay, output_name, "Precision-Recall Curve", ax)
