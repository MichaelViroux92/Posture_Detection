import os
import mlflow
import mlflow.xgboost
import xgboost as xgb

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, ConfusionMatrixDisplay,
    RocCurveDisplay, PrecisionRecallDisplay
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import RandomizedSearchCV


class XGBoostBaseModelRunner:
    def __init__(self, target, X_train, y_train, X_val, y_val, base_params=None):
        self.target = target
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train[target]
        self.y_val = y_val[target]
        self.n_classes = self.y_train.nunique()

        self.base_params = base_params or {
            "n_estimators": 300,
            "learning_rate": 0.1,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "gamma": 0,
            "reg_lambda": 1,
            "reg_alpha": 0,
            "use_label_encoder": False
        }

        if self.n_classes == 2:
            self.base_params.update({"objective": "binary:logistic", "eval_metric": "logloss"})
        else:
            self.base_params.update({"objective": "multi:softprob", "num_class": self.n_classes, "eval_metric": "mlogloss"})

        self.model = xgb.XGBClassifier(**self.base_params)
        self.metrics = {}

    def train(self):
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self):
        y_pred = self.model.predict(self.X_val)
        y_prob = self.model.predict_proba(self.X_val)
        avg = "binary" if self.n_classes == 2 else "weighted"

        self.metrics = {
            "accuracy": accuracy_score(self.y_val, y_pred),
            "precision": precision_score(self.y_val, y_pred, average=avg, zero_division=0),
            "recall": recall_score(self.y_val, y_pred, average=avg, zero_division=0),
            "f1": f1_score(self.y_val, y_pred, average=avg, zero_division=0),
            "roc_auc": roc_auc_score(self.y_val, y_prob[:, 1]) if self.n_classes == 2 else roc_auc_score(self.y_val, y_prob, multi_class='ovr'),
            "log_loss": log_loss(self.y_val, y_prob)
        }
        return self.metrics

    def log_to_mlflow(self, experiment_name=None):
        if experiment_name:
            mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=f"xgboost_{self.target}"):
            mlflow.set_tag("target", self.target)
            mlflow.log_params(self.base_params)
            mlflow.log_metrics(self.metrics)
            mlflow.xgboost.log_model(self.model, artifact_path="model")

    def plot_confusion_matrix(self, ax):
        y_pred = self.model.predict(self.X_val)
        ConfusionMatrixDisplay.from_predictions(self.y_val, y_pred, ax=ax, colorbar=False)
        ax.set_title(f"Confusion Matrix - {self.target}")

    def plot_feature_importance(self, ax):
        xgb.plot_importance(self.model, max_num_features=10, ax=ax)
        ax.set_title(f"Feature Importance - {self.target}")

    def _plot_curve(self, display_fn, title, ax):
        y_prob = self.model.predict_proba(self.X_val)
        y_true = self.y_val

        if self.n_classes == 2:
            display_fn.from_predictions(y_true, y_prob[:, 1], ax=ax)
        else:
            y_true_bin = label_binarize(y_true, classes=np.arange(self.n_classes))
            for i in range(self.n_classes):
                display_fn.from_predictions(
                    y_true_bin[:, i], y_prob[:, i], name=f"Class {i}", ax=ax
                )

        ax.set_title(f"{title} - {self.target}")
    
    def plot_roc_curve(self, ax):
        self._plot_curve(RocCurveDisplay, "ROC Curve", ax)

    def plot_precision_recall(self, ax):
        self._plot_curve(PrecisionRecallDisplay, "Precision-Recall Curve", ax)



class XGBoostTunedModelRunner(XGBoostBaseModelRunner):
    def __init__(self, target, X_train, y_train, X_val, y_val, param_grid=None, scoring=None, base_params=None):
        super().__init__(target, X_train, y_train, X_val, y_val, base_params)
        self.param_grid = param_grid
        self.scoring = scoring
        self.best_model = None

    def tune(self, n_iter=20, cv=3, n_jobs=-1, random_state=42):
        """
        Run RandomizedSearchCV to find best hyperparameters.
        """
        model = clone(self.model)  # clone base model to not mess original

        grid = RandomizedSearchCV(
            model,
            param_distributions=self.param_grid,
            scoring=self.scoring,
            n_iter=n_iter,
            cv=cv,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=1
        )
        grid.fit(self.X_train, self.y_train)
        self.best_model = grid.best_estimator_

        print(f"Best params for {self.target}: {grid.best_params_}")

        # Replace the model with best one for subsequent methods
        self.model = self.best_model

    def log_to_mlflow(self, experiment_name=None):
        if experiment_name:
            mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=f"xgboost_{self.target}"):
            mlflow.set_tag("target", self.target)
            mlflow.log_params(self.base_params)
            mlflow.log_metrics(self.metrics)
            mlflow.xgboost.log_model(self.model, artifact_path="model")
            mlflow.log_params({f"tuned__{k}": v for k, v in self.best_model.get_params().items()})
            mlflow.log_dict(self.param_grid, "param_grid.json")
            mlflow.log_dict(self.scoring, "scoring_grid.json")





    



    