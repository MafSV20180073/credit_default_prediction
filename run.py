import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier

from modeling.model_trainer import ModelTrainer
from preprocessing.data_loader import download_data_from_kaggle
from preprocessing.data_processor import DataProcessor
from preprocessing.processing_steps import split_dataset

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    seed = 17
    random_state = np.random.RandomState(seed)
    mappings = {
        "education": {
            1: 1,  # "Graduate School",
            2: 2,  # "University",
            3: 3,  # "High School",
            4: 4,  # "Others",
            5: 4,  # "Unknown",
            6: 4,  # "Unknown"
            0: 4,  # "Unknown"
        }
    }
    download_from_kaggle = False
    suffix, path = "treated_v2", "data/"

    if download_from_kaggle:
        df = download_data_from_kaggle()
        X, y = df.iloc[:, :-1], df.iloc[:, -1]
        splits = split_dataset(
            X=X, y=y, random_state=random_state, shuffle=True, stratify=True
        )

        data_processor = DataProcessor(
            splits=splits, mappings=mappings, data_from_kaggle=True
        )
        # Deal with categorical variables:
        data_processor.treat_categorical_variables(drop_original_vars=True)

        # Perform feature engineering:
        bill_amt_cols = [
            "bill_amt1",
            "bill_amt2",
            "bill_amt3",
            "bill_amt4",
            "bill_amt5",
            "bill_amt6",
        ]
        pay_amt_cols = [
            "pay_amt1",
            "pay_amt2",
            "pay_amt3",
            "pay_amt4",
            "pay_amt5",
            "pay_amt6",
        ]
        list_vars_to_drop = bill_amt_cols + pay_amt_cols

        data_processor.feature_engineering(
            calculate_bill_to_limit_bal_ratio=True,
            calculate_pay_to_bill_ratio=True,
            calculate_num_negative_bill_statements=True,
            calculate_payment_delays=True,
            calculate_payment_change_rate=True,
            calculate_bill_change_rate=True,
            calculate_total_payment=True,
            list_vars_to_drop=list_vars_to_drop,
        )

        # Deal with outliers:
        data_processor.treat_outliers(verbose=True)

        # Normalize data:
        # data_processor.standardize_data()

        # 'treated_v1' includes categorical variables treatment, feature engineering (with drop of original variables) and outlier treatment
        # 'treated_v2' includes categorical variables treatment, feature engineering (with drop of original variables), outlier treatment and normalization
        data_processor.export_datasets(suffix=suffix, path=path)

    else:
        splits = {}
        train_data, val_data, test_data = (
            pd.read_csv(f"{path}train_set_{suffix}.csv"),
            pd.read_csv(f"{path}val_set_{suffix}.csv"),
            pd.read_csv(f"{path}test_set_{suffix}.csv"),
        )
        splits["X_train"], splits["y_train"] = (
            train_data.iloc[:, :-1],
            train_data.iloc[:, -1],
        )
        splits["X_val"], splits["y_val"] = val_data.iloc[:, :-1], val_data.iloc[:, -1]
        splits["X_test"], splits["y_test"] = (
            test_data.iloc[:, :-1],
            test_data.iloc[:, -1],
        )

        data_processor = DataProcessor(
            splits=splits, mappings=mappings, data_from_kaggle=False
        )

    X_train, y_train, X_val, y_val, X_test, y_test = data_processor.get_datasets()

    model_trainer = ModelTrainer(
        X_train, y_train, X_val, y_val, X_test, y_test, seed=random_state
    )

    abc = AdaBoostClassifier(random_state=seed)

    model_trainer.train_classifier(
        classifier=abc,
        classifier_name="AdaBoost",
        perform_feature_selection=False,
        feature_selection_algorithm="f_classif",
        num_features=15,
    )

    model_trainer.evaluate_classifier(
        classifier_name="AdaBoost",
        store_results=True,
        print_classification_report=True,
    )

    # TODO: review feature selection (see SelectFromModel and SelectKBest)
    # TODO: hyperparameter tuning
    # TODO: fairgbm

    stop_here = True
