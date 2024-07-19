import os
import pickle
from datetime import datetime
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


class DelayModel:
    FEATURES_COLS = [
        "MES_4",
        "MES_7",
        "MES_10",
        "MES_11",
        "MES_12",
        "OPERA_Copa Air",
        "OPERA_Grupo LATAM",
        "OPERA_Latin American Wings",
        "OPERA_Sky Airline",
        "TIPOVUELO_I",
    ]

    THRESHOLD_IN_MINUTES = 15

    def __init__(self):
        self._model = LogisticRegression()

    def _get_min_diff(self, data: pd.Series) -> float:
        """
        Auxiliary function to get target.

        Args:
            data (pd.Series): raw data row.

        Returns:
            float: difference between two rows in minutes.
        """
        fecha_o = datetime.strptime(data["Fecha-O"], "%Y-%m-%d %H:%M:%S")
        fecha_i = datetime.strptime(data["Fecha-I"], "%Y-%m-%d %H:%M:%S")
        min_diff = ((fecha_o - fecha_i).total_seconds()) / 60
        return min_diff

    def _get_delay_target(self, data: pd.DataFrame) -> pd.Series:
        """
        Compute and return target to train the model with, from raw data.

        Args:
            data (pd.DataFrame): raw data.

        Returns:
            pd.Series: target to predict.
        """
        data["min_diff"] = data.apply(self._get_min_diff, axis=1)
        data["delay"] = np.where(data["min_diff"] > self.THRESHOLD_IN_MINUTES, 1, 0)

        return data["delay"].to_frame()

    def _get_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute and return input features to feed the model from raw data.

        Args:
            data (pd.DataFrame): raw_data.

        Returns:
            pd.DataFrame: features with columns in a specific order.
        """
        # get the one hot enconding of the columns suggested by the DS
        # the existance of these three columns is enforced by the api above this code
        features = pd.concat(
            [
                pd.get_dummies(data["OPERA"], prefix="OPERA"),
                pd.get_dummies(data["TIPOVUELO"], prefix="TIPOVUELO"),
                pd.get_dummies(data["MES"], prefix="MES"),
            ],
            axis=1,
        )
        valid_features = list(
            set(self.FEATURES_COLS).intersection(set(features.columns))
        )
        missing_features = list(
            set(self.FEATURES_COLS).difference(set(features.columns))
        )

        # get valid features and fill missin with  0 due to one-hot encoding
        features = features[valid_features]
        features[missing_features] = 0

        # return dataframe with sorted columns
        return features[self.FEATURES_COLS]

    def preprocess(
        self, data: pd.DataFrame, target_column: Optional[str] = None
    ) -> Union[Tuple[pd.DataFrame, pd.Series], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        # retrieve features from the data
        x = self._get_features(data)

        # return different sets, depending on the target
        if target_column is None:
            return x
        elif target_column == "delay":
            y = self._get_delay_target(data)
            return (x, y)
        else:
            raise NotImplementedError("Only implemented 'delay' as target column")

    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """
        Fit model with data preprocessed by this class.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        # target is assume to only have one column (uni-dimensional target)
        target_column = target.columns[0]

        # get values to compensate unbalancing
        n_y0 = len(target[target[target_column] == 0])
        n_y1 = len(target[target[target_column] == 1])

        # instantiate model and fit
        self._model = LogisticRegression(
            class_weight={1: n_y0 / len(target), 0: n_y1 / len(target)}
        )
        self._model.fit(features, target[target_column])

        # for scikitlearn compatibility
        return self

    def save_model(self, path: str):
        """
        Store trained model in given path
        """
        os.makedirs(path, exist_ok=True)
        with open(f"{path}/model.pkl", "wb") as f:
            pickle.dump(self._model, f)

    def load_model(self, path: str):
        """
        Load trained model from given path
        """
        with open(f"{path}/model.pkl", "rb") as f:
            self._model = pickle.load(f)

    def predict(self, features: pd.DataFrame) -> List[int]:
        """
        Predict delays for new flights on data preprocessed by this class.

        Args:
            features (pd.DataFrame): preprocessed data.

        Returns:
            (List[int]): predicted targets.
        """
        try:
            pred = self._model.predict(features).tolist()
        except Exception as e:
            raise e

        return pred


def main():
    # perform a training of the model with all available data for production deployment

    # get data and initial model
    model = DelayModel()
    data = pd.read_csv(filepath_or_buffer="data/data.csv")

    # preprocess data and fit
    features, target = model.preprocess(data=data, target_column="delay")
    model.fit(features=features, target=target)

    # save
    model.save_model("models")


if __name__ == "__main__":
    main()
