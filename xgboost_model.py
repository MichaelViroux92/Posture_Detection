# XGboost model

#--------- PARAMETERS GRID ---------

PARAMETERS_GRID = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 6, 9],
    "learning_rate": [0.01, 0.1, 0.2]
}


import numpy as np
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
import numpy as np

class XGBoostTrainer:
    def __init__(self, df_features_targets):
        """
        Initializes the training class.

        Parameters:
        - df_features_targets (pd.DataFrame): Dataframe with features and target columns.
        """
        self.df = df_features_targets
        self.model = None
        self.X_train, self.X_test, self.y_train, self.y_test = self.preprocess_data()
        self.best_params = None  # Store best hyperparameters
        self.param_grid = PARAMETERS_GRID

    def preprocess_data(self, target_column):
        """Prepare features (X) and targets (y), and split into train/test."""
        X = self.df.drop(target_column, axis=1)
        y = self.df[target_column]

        return train_test_split(X, y, test_size=0.2, random_state=42)

    def hyperparameter_tuning(self):
        """
        Perform hyperparameter tuning using GridSearchCV with MultiOutputRegressor.
        """
        param_grid_names = {f"estimator__{key}": value for key, value in self.param_grid.items()}
        grid_search = GridSearchCV(
            estimator=xgb.XGBRegressor(),
            param_grid=param_grid_names,
            cv=3,
            scoring="neg_mean_squared_error",
            verbose=1,
            n_jobs=-1
        )

        grid_search.fit(self.X_train, self.y_train)

        self.best_params = {key.replace("estimator__", ""): value for key, value in grid_search.best_params_.items()}
        return {"best_params": self.best_params, "best_score": grid_search.best_score_}

    def train_model(self):
        """Train an XGBoost model using the best parameters from GridSearch."""
        if not self.best_params:
            raise ValueError("Run hyperparameter_tuning() first to get best parameters.")

        self.model = xgb.XGBRegressor(**self.best_params) # Use best params
        self.model.fit(self.X_train, self.y_train)
        self.model.save_model("model.json")

    def evaluate_model(self):
        """Evaluate model performance using MAE and RMSE."""
        y_pred = self.model.predict(self.X_test)
        mae = mean_absolute_error(self.y_test, y_pred)
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)

        return {"MAE": mae, "RMSE": rmse}
    

def predict_from_model(model_path, X):
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    return model.predict(X)