import lightgbm as lgb
import pandas as pd


class ClickModel:
    def __init__(self, model_path: str | None = None):
        # Load ONCE here, never inside predict_batch
        self.model = lgb.Booster(model_file=model_path) if model_path else None

    def predict_batch(self, features: pd.DataFrame):
        # Whole batch in one call; never loop row by row
        if self.model is None:
            raise RuntimeError("No model loaded")
        return self.model.predict(features)

    def save(self, path: str) -> None:
        self.model.save_model(path)
