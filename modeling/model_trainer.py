import numpy as np

from sklearn.feature_selection import f_classif, mutual_info_classif, SelectKBest
from sklearn.metrics import balanced_accuracy_score, classification_report, f1_score, recall_score, precision_score
from sklearn.model_selection import HalvingRandomSearchCV
from typing import Dict, Union


class ModelTrainer:
    """
    Class that aggregates a set of functions to perform feature selection, train and evaluate the models and store the results.
    Notice that all classifiers used must be compatible with scikit-learn API. This class also requires the passing
    of three datasets: training, validation and test.

    Parameters
    ----------

    """


    def __init__(
            self,
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            seed
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test

        self._seed = seed

        self.models = {}
        self.models_results = {}
        self.models_metadata = {}
        self.models_hyperparameter_tuning = {}

        # Variables to be initialized later:
        self.feature_selector = None


    def train_classifier(
            self,
            classifier: object,
            classifier_name: str,
            perform_feature_selection: bool = False,
            feature_selection_algorithm: str = None,
            num_features: int = 20,
    ):
        """ """
        if perform_feature_selection:
            self.perform_feature_selection(algorithm=feature_selection_algorithm, num_features=num_features)

        classifier.fit(self.X_train, self.y_train)

        # Store the trained model:
        self.models[classifier_name] = classifier


    def evaluate_classifier(
            self,
            classifier_name: str,
            store_results: bool = True,
            print_classification_report: bool = True
    ):
        """ """
        def calculate_metrics(y_true, y_preds, prefix):
            return {
                f"{prefix}_recall_score": round(recall_score(y_true, y_preds), 2),
                f"{prefix}_precision_score": round(precision_score(y_true, y_preds), 2),
                f"{prefix}_f1_score": round(f1_score(y_true, y_preds), 2),
                f"{prefix}_balanced_accuracy_score": round(balanced_accuracy_score(y_true, y_preds), 2)
            }


        assert classifier_name in self.models, f"There is no model registered under the name {classifier_name}."

        classifier = self.models.get(classifier_name)

        train_preds = classifier.predict(self.X_train)
        val_preds = classifier.predict(self.X_val)
        test_preds = classifier.predict(self.X_test)

        scores_dict = {}

        scores_dict.update(calculate_metrics(self.y_test, test_preds, "test"))
        scores_dict.update(calculate_metrics(self.y_val, val_preds, "val"))
        scores_dict.update(calculate_metrics(self.y_train, train_preds, "train"))

        # Store the results in the model trainer:
        if store_results:
            self.models_results[classifier_name] = scores_dict

        if print_classification_report:
            print(f"\t#### {classifier_name.upper()} RESULTS ####")
            print("Train scores:")
            print(classification_report(self.y_train.values, train_preds))
            print("Val scores:")
            print(classification_report(self.y_val.values, val_preds))
            print("Test scores:")
            print(classification_report(self.y_test.values, test_preds))

        return scores_dict


    def perform_feature_selection(self, algorithm: str = "mutual_info_classif", num_features: int = 20):
        """Performs feature selection by applying 'SelectKBest' from scikit-learn library. It's prepared to use 1 of 2
        score functions: mutual_info_classif and f_classif."""
        assert algorithm in ["f_classif", "mutual_info_classif"], f"Expected 'mutual_info_classif' or 'f_classif', got {algorithm} instead."

        if algorithm == "mutual_info_classif":
            self.feature_selector = SelectKBest(mutual_info_classif, k=num_features)

        else:
            self.feature_selector = SelectKBest(f_classif, k=num_features)

        self.feature_selector.fit(self.X_train, self.y_train)

        print("Feature selection result:\n", self.feature_selector.get_feature_names_out())

        self.X_train = self.X_train[self.feature_selector.get_feature_names_out()]
        self.X_val = self.X_val[self.feature_selector.get_feature_names_out()]
        self.X_test = self.X_test[self.feature_selector.get_feature_names_out()]


    def perform_hyperparameter_tuning(
            self,
            classifier: object,
            classifier_name: str,
            param_distributions: Dict,
            resource: str = "n_estimators",
            max_resources: int = 10,
            random_state: Union[int, np.random.RandomState] = 17,
            refit: bool = True
    ):
        """ """
        search = HalvingRandomSearchCV(
                estimator=classifier,
                param_distributions=param_distributions,
                resource=resource,
                max_resources=max_resources,
                random_state=random_state,
                refit=refit
        )

        search.fit(self.X_train, self.y_train)

        self.models_hyperparameter_tuning[classifier_name] = {
            "best_params": search.best_params_,
            "best_val_score": search.score(self.X_val, self.y_val),
            "cv_results": search.cv_results_
        }


    def add_classifier_metadata(self, classifier_name: str, metadata: Union[Dict, str]):
        """Allows to store some additional information about the model, namely used params, structure of the pipeline
        used to treat and prepare the data, among others."""
        self.models_metadata[classifier_name] = metadata


    def get_datasets(self):
        """Returns the three datasets split into X and y."""
        return self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test


    def get_results_dict(self):
        """Returns the dictionary containing the results for the trained models."""
        return self.models_results


    def reload_data(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """Reloads the original sets for train, val and test."""
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test


