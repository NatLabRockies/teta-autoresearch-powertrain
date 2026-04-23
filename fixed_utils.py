import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Fixed Data utilities (do not modify)
# ---------------------------------------------------------------------------


def train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train/test sets."""
    rng = np.random.default_rng(random_seed)
    mask = rng.random(len(df)) < test_size
    test_df = df[mask].reset_index(drop=True)
    train_df = df[~mask].reset_index(drop=True)
    return train_df, test_df


# ---------------------------------------------------------------------------
# Evaluation Metrics (fixed, do not modify)
# ---------------------------------------------------------------------------


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def evaluate(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:
    """Evaluate a trained model on test data and return all metrics."""
    return {
        "rmse": rmse(actual, predicted),
    }
