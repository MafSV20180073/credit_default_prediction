from typing import Dict, List

import numpy as np
import pandas as pd
import scipy as sp
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler


class DataProcessor:
    """
    Class that aggregates a set of functions for loading, cleaning,
    feature engineering, encoding, and normalize data, among others,
    with the goal of preparing the data to be feed into a model.

    Parameters
    ----------
    target_col_name : str, optional
        Name of the target variable, default is 'target'.

    val_size : float, optional
        Represents the proportion of the train dataset to include in the
        validation split, by default is 0.2.
    """

    def __init__(
        self,
        splits: Dict,
        target_col_name: str = "target",
        data_from_kaggle: bool = False,
    ):
        assert all(
            key in splits
            for key in ["X_train", "y_train", "X_test", "y_test", "X_val", "y_val"]
        ), (
            "Splits argument must contain X_train, y_train, X_test, y_test, "
            "X_val, y_val keys."
        )
        self.X_train = splits.get("X_train")
        self.y_train = splits.get("y_train")
        self.X_test = splits.get("X_test")
        self.y_test = splits.get("y_test")
        self.X_val = splits.get("X_val")
        self.y_val = splits.get("y_val")

        if data_from_kaggle:
            self._initial_dataset_uniformization("train")
            self._initial_dataset_uniformization("test")
            self._initial_dataset_uniformization("val")

        self.target_name = target_col_name

        # Variables to be initialized later:
        self.standard_scaler = None
        self.transformer = None
        self._list_vars_to_drop = None
        self._calculate_total_payment = None
        self._calculate_bill_change_rate = None
        self._calculate_payment_change_rate = None
        self._calculate_payment_delays = None
        self._calculate_num_negative_bill_statements = None
        self._calculate_pay_to_bill_ratio = None
        self._calculate_bill_to_limit_bal_ratio = None

    def treat_categorical_variables(self, drop_original_vars: bool = True):
        """Encoding of 'education' variable and creation of flags for clients
        who are male, married, or single."""
        self.X_train = self._treat_categorical_variables(
            self.X_train, drop_original_vars
        )
        self.X_val = self._treat_categorical_variables(self.X_val, drop_original_vars)
        self.X_test = self._treat_categorical_variables(self.X_test, drop_original_vars)

    def feature_engineering(
        self,
        calculate_bill_to_limit_bal_ratio: bool = False,
        calculate_pay_to_bill_ratio: bool = False,
        calculate_num_negative_bill_statements: bool = False,
        calculate_payment_delays: bool = False,
        calculate_payment_change_rate: bool = False,
        calculate_bill_change_rate: bool = False,
        calculate_total_payment: bool = False,
        list_vars_to_drop: List = None,
    ):
        """Performs feature engineering on the given dataset and returns the dataset with additional
        engineered features."""
        self._calculate_bill_to_limit_bal_ratio = calculate_bill_to_limit_bal_ratio
        self._calculate_pay_to_bill_ratio = calculate_pay_to_bill_ratio
        self._calculate_num_negative_bill_statements = (
            calculate_num_negative_bill_statements
        )
        self._calculate_payment_delays = calculate_payment_delays
        self._calculate_payment_change_rate = calculate_payment_change_rate
        self._calculate_bill_change_rate = calculate_bill_change_rate
        self._calculate_total_payment = calculate_total_payment
        self._list_vars_to_drop = list_vars_to_drop

        # Perform feature engineering:
        self.X_train = self._feature_engineering(self.X_train)
        self.X_val = self._feature_engineering(self.X_val)
        self.X_test = self._feature_engineering(self.X_test)

    def standardize_data(self, column_names: List, exclude_column_names: bool = True):
        """Standardize features by removing the mean and scaling to unit variance."""
        columns_to_standardize = [col for col in self.X_train.columns if col not in column_names] if exclude_column_names else column_names

        self.transformer = ColumnTransformer(
            transformers=[("", StandardScaler(), columns_to_standardize)],
            remainder="passthrough",
        )

        X_train_std = self.transformer.fit_transform(self.X_train[columns_to_standardize])
        self.X_train[columns_to_standardize] = X_train_std

        X_val_std = self.transformer.transform(self.X_val[columns_to_standardize])
        self.X_val[columns_to_standardize] = X_val_std  # pd.DataFrame(X_val_std, columns=col_names, index=self.y_val.index)

        X_test_std = self.transformer.transform(self.X_test[columns_to_standardize])
        self.X_test[columns_to_standardize] = X_test_std

    def treat_outliers(self, column_names: List, exclude_column_names: bool = True, verbose: bool = False):
        """Identifies and treats outliers using the interquartilic range criteria."""
        def detect_outliers(
            data: pd.DataFrame, col: str, iqr_cut_point: float, dataset_type: str
        ):
            """Detects outliers in a column given the IQR cut point."""
            out = data[(data[col] > iqr_cut_point) | (data[col] < -iqr_cut_point)]
            if len(out):
                if verbose:
                    print(
                        f"Number of outliers detected in '{col}' column of {dataset_type} "
                        f"dataset: {len(out)} ({round(len(out)/len(data) * 100,3)}% of dataset "
                        f"rows with values below or above 1.5 IQR from 1Q/3Q)."
                    )
                return True

            return False

        def remove_outliers(data: pd.DataFrame, col: str, iqr_cut_point: float):
            """Replaces outlier values with the IQR cut point."""
            data[col] = data[col].apply(
                lambda x: iqr_cut_point
                if x > iqr_cut_point
                else (-iqr_cut_point if x < -iqr_cut_point else x)
            )
            return data

        numeric_columns = [col for col in self.X_train.columns if col not in column_names] if exclude_column_names else column_names

        for column in numeric_columns:
            out_cut = np.nanpercentile(self.X_train[column], 75) + (
                sp.stats.iqr(self.X_train[column], nan_policy="omit") * 1.5
            )

            if detect_outliers(self.X_train, column, out_cut, "train"):
                self.X_train = remove_outliers(self.X_train, column, out_cut)

            if detect_outliers(self.X_val, column, out_cut, "validation"):
                self.X_val = remove_outliers(self.X_val, column, out_cut)

            if detect_outliers(self.X_test, column, out_cut, "test"):
                self.X_test = remove_outliers(self.X_test, column, out_cut)

    def get_datasets(self):
        """Returns the three datasets split into X and y."""
        return (
            self.X_train,
            self.y_train,
            self.X_val,
            self.y_val,
            self.X_test,
            self.y_test,
        )

    def export_datasets(self, suffix: str, path: str = "../data"):
        """Exports the three datasets as they currently are."""
        train_data = pd.concat(
            [self.X_train, pd.DataFrame({"target": self.y_train})], axis=1
        )
        val_data = pd.concat([self.X_val, pd.DataFrame({"target": self.y_val})], axis=1)
        test_data = pd.concat(
            [self.X_test, pd.DataFrame({"target": self.y_test})], axis=1
        )

        train_data.to_csv(f"{path}/train_set_{suffix}.csv", index=False)
        val_data.to_csv(f"{path}/val_set_{suffix}.csv", index=False)
        test_data.to_csv(f"{path}/test_set_{suffix}.csv", index=False)

        del train_data
        del val_data
        del test_data

    def reload_datasets(self, splits: Dict):
        """Reloads the three datasets."""
        assert all(
            key in splits
            for key in ["X_train", "y_train", "X_test", "y_test", "X_val", "y_val"]
        ), (
            "Splits argument must contain X_train, y_train, X_test, y_test, "
            "X_val, y_val keys."
        )

        self.X_train = splits.get("X_train")
        self.y_train = splits.get("y_train")
        self.X_test = splits.get("X_test")
        self.y_test = splits.get("y_test")
        self.X_val = splits.get("X_val")
        self.y_val = splits.get("y_val")

    def _map_values(self, col, mapping):
        """Function to encode a column in a dataframe."""
        return col.map(mapping)

    def _initial_dataset_uniformization(self, dataset_type: str):
        """Performs initial steps in the dataset such as renaming columns
        and removing 'id' column."""
        rename_cols_map = {"PAY_0": "PAY_1"}

        if dataset_type == "train":
            self.X_train.rename(columns=rename_cols_map, inplace=True)
            self.X_train.columns = [col.lower() for col in self.X_train.columns]
            self.X_train.drop(columns=["id"], axis=1, inplace=True)
        elif dataset_type == "test":
            self.X_test.rename(columns=rename_cols_map, inplace=True)
            self.X_test.columns = [col.lower() for col in self.X_test.columns]
            self.X_test.drop(columns=["id"], axis=1, inplace=True)
        elif dataset_type == "val":
            self.X_val.rename(columns=rename_cols_map, inplace=True)
            self.X_val.columns = [col.lower() for col in self.X_val.columns]
            self.X_val.drop(columns=["id"], axis=1, inplace=True)
        else:
            raise ValueError(
                f"dataset_type arg must be 'train', 'test' or 'val', got {dataset_type} instead."
            )

    def _treat_categorical_variables(
        self, data: pd.DataFrame, drop_original_vars: bool = True
    ):
        """Internal function that reencodes of 'education' variable and creation
        of flags for clients who are male, married, or single."""
        data["is_male"] = np.where(data.sex == 1, 1, 0)
        data["is_married"] = np.where(data.marriage == 1, 1, 0)
        data["is_single"] = np.where(data.marriage == 2, 1, 0)
        data["graduate_school_education"] = np.where(data.education == 1, 1, 0)
        data["university_education"] = np.where(data.education == 2, 1, 0)
        data["high_school_education"] = np.where(data.education == 3, 1, 0)

        if drop_original_vars:
            data.drop(["sex", "marriage", "education"], axis=1, inplace=True)

        return data

    def _feature_engineering(self, data: pd.DataFrame):
        """Performs feature engineering on the given dataset and returns the dataset with additional
        engineered features."""

        def avoid_zero_division(row, pay_amt, bill_amt):
            """Utility function that returns the pay_amtX when bill_amtX is equal to zero."""
            if row[bill_amt] == 0:
                # If pay_amt is positive, it means the client has overpaid:
                return -row[pay_amt]

            return 0.0 if row[pay_amt] == 0 else row[pay_amt] / row[bill_amt]

        def calculate_change_rate(row, col_amt1, col_amt2):
            """Calculates the change rate (pay_amt or bill_amt) between two consecutive months."""
            if row[col_amt1] == 0:
                return 0.0

            return (row[col_amt2] - row[col_amt1]) / row[col_amt1]

        if self._calculate_bill_to_limit_bal_ratio:
            data["bill_amt1_limit_bal_ratio"] = data.bill_amt1 / data.limit_bal
            data["bill_amt2_limit_bal_ratio"] = data.bill_amt2 / data.limit_bal
            data["bill_amt3_limit_bal_ratio"] = data.bill_amt3 / data.limit_bal
            data["bill_amt4_limit_bal_ratio"] = data.bill_amt4 / data.limit_bal
            data["bill_amt5_limit_bal_ratio"] = data.bill_amt5 / data.limit_bal
            data["bill_amt6_limit_bal_ratio"] = data.bill_amt6 / data.limit_bal

        if self._calculate_pay_to_bill_ratio:
            data["pay_amt1_bill_amt1_ratio"] = data.apply(
                lambda x: avoid_zero_division(x, "pay_amt1", "bill_amt1"), axis=1
            )
            data["pay_amt2_bill_amt2_ratio"] = data.apply(
                lambda x: avoid_zero_division(x, "pay_amt2", "bill_amt2"), axis=1
            )
            data["pay_amt3_bill_amt3_ratio"] = data.apply(
                lambda x: avoid_zero_division(x, "pay_amt3", "bill_amt3"), axis=1
            )
            data["pay_amt4_bill_amt4_ratio"] = data.apply(
                lambda x: avoid_zero_division(x, "pay_amt4", "bill_amt4"), axis=1
            )
            data["pay_amt5_bill_amt5_ratio"] = data.apply(
                lambda x: avoid_zero_division(x, "pay_amt5", "bill_amt5"), axis=1
            )
            data["pay_amt6_bill_amt6_ratio"] = data.apply(
                lambda x: avoid_zero_division(x, "pay_amt6", "bill_amt6"), axis=1
            )

        if self._calculate_num_negative_bill_statements:
            bill_amt_cols = [
                "bill_amt1",
                "bill_amt2",
                "bill_amt3",
                "bill_amt4",
                "bill_amt5",
                "bill_amt6",
            ]
            data["num_overpays"] = (data[bill_amt_cols] < 0).sum(axis=1)

        if self._calculate_payment_delays:
            data["payment_delay_amt1"] = (data.bill_amt1 - data.pay_amt1).apply(
                lambda x: max(0, x)
            )
            data["payment_delay_amt2"] = (data.bill_amt2 - data.pay_amt2).apply(
                lambda x: max(0, x)
            )
            data["payment_delay_amt3"] = (data.bill_amt3 - data.pay_amt3).apply(
                lambda x: max(0, x)
            )
            data["payment_delay_amt4"] = (data.bill_amt4 - data.pay_amt4).apply(
                lambda x: max(0, x)
            )
            data["payment_delay_amt5"] = (data.bill_amt5 - data.pay_amt5).apply(
                lambda x: max(0, x)
            )
            data["payment_delay_amt6"] = (data.bill_amt6 - data.pay_amt6).apply(
                lambda x: max(0, x)
            )

        if self._calculate_payment_change_rate:
            data["payment_change_rate_amt1_amt2"] = data.apply(
                lambda x: calculate_change_rate(x, "pay_amt1", "pay_amt2"), axis=1
            )
            data["payment_change_rate_amt2_amt3"] = data.apply(
                lambda x: calculate_change_rate(x, "pay_amt2", "pay_amt3"), axis=1
            )
            data["payment_change_rate_amt3_amt4"] = data.apply(
                lambda x: calculate_change_rate(x, "pay_amt3", "pay_amt4"), axis=1
            )
            data["payment_change_rate_amt4_amt5"] = data.apply(
                lambda x: calculate_change_rate(x, "pay_amt4", "pay_amt5"), axis=1
            )
            data["payment_change_rate_amt5_amt6"] = data.apply(
                lambda x: calculate_change_rate(x, "pay_amt5", "pay_amt6"), axis=1
            )

        if self._calculate_bill_change_rate:
            data["bill_change_rate_amt1_amt2"] = data.apply(
                lambda x: calculate_change_rate(x, "bill_amt1", "bill_amt2"), axis=1
            )
            data["bill_change_rate_amt2_amt3"] = data.apply(
                lambda x: calculate_change_rate(x, "bill_amt2", "bill_amt3"), axis=1
            )
            data["bill_change_rate_amt3_amt4"] = data.apply(
                lambda x: calculate_change_rate(x, "bill_amt3", "bill_amt4"), axis=1
            )
            data["bill_change_rate_amt4_amt5"] = data.apply(
                lambda x: calculate_change_rate(x, "bill_amt4", "bill_amt5"), axis=1
            )
            data["bill_change_rate_amt5_amt6"] = data.apply(
                lambda x: calculate_change_rate(x, "bill_amt5", "bill_amt6"), axis=1
            )

        if self._calculate_total_payment:
            pay_amt_cols = [
                "pay_amt1",
                "pay_amt2",
                "pay_amt3",
                "pay_amt4",
                "pay_amt5",
                "pay_amt6",
            ]
            data["total_payment"] = (data[pay_amt_cols]).sum(axis=1)

        if self._list_vars_to_drop and len(self._list_vars_to_drop) > 0:
            data.drop(self._list_vars_to_drop, axis=1, inplace=True)

        return data
