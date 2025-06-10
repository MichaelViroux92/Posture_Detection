import pandas as pd
from ml_models.xgboost_model import XGBoostBaseModelRunner, XGBoostTunedModelRunner

def collect_all_metrics(runners):
    metric_rows = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC", "Logloss"]
    target_columns = list(runners.keys())
    df_metrics = pd.DataFrame(index=metric_rows, columns=target_columns)

    for target, runner in runners.items():
        runner.evaluate()  # ensures runner.metrics is populated
        m = runner.metrics

        df_metrics.loc["Accuracy", target] = m["accuracy"]
        df_metrics.loc["Precision", target] = m["precision"]
        df_metrics.loc["Recall", target] = m["recall"]
        df_metrics.loc["F1 Score", target] = m["f1"]
        df_metrics.loc["ROC AUC", target] = m["roc_auc"]
        df_metrics.loc["Logloss", target] = m["log_loss"]

    return df_metrics