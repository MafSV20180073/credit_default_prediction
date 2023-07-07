from typing import Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataset(
    X: pd.DataFrame,
    y: pd.DataFrame,
    test_size: float = 0.15,
    generate_validation_data: bool = True,
    val_size: float = 0.15,
    random_state: Union[int, np.random.RandomState] = 17,
    shuffle: bool = False,
    stratify: bool = False,
):
    """Splits a dataset into training, validation and test sets. This function
    returns a dictionary containing the requested splits."""
    splits = {}
    original_size = len(X)

    if not stratify:
        stratify = None
    else:
        stratify = y

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        shuffle=shuffle,
        random_state=random_state,
        stratify=stratify,
    )

    splits["X_train"] = X_train
    splits["y_train"] = y_train
    splits["X_test"] = X_test
    splits["y_test"] = y_test

    print(
        f"Training/test split created: {len(X_train)} training samples /{len(X_test)} test samples."
    )

    if generate_validation_data:
        num_val_samples = int(round((original_size * val_size), 0))

        if stratify is not None:
            stratify = y_train

        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=num_val_samples,
            shuffle=True,
            random_state=random_state,
            stratify=stratify,
        )

        splits["X_train"] = X_train
        splits["y_train"] = y_train
        splits["X_val"] = X_val
        splits["y_val"] = y_val

        print(
            f"Training/validation split created: {len(X_train)} training "
            f"samples /{len(X_val)} val samples."
        )

    return splits
