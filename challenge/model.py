from datetime import datetime
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

FEATURES_COLS = [
    "OPERA_Latin American Wings",
    "MES_7",
    "MES_10",
    "OPERA_Grupo LATAM",
    "MES_12",
    "TIPOVUELO_I",
    "MES_4",
    "MES_11",
    "OPERA_Sky Airline",
    "OPERA_Copa Air",
]


def get_min_diff(data):
    fecha_o = datetime.strptime(data["Fecha-O"], "%Y-%m-%d %H:%M:%S")
    fecha_i = datetime.strptime(data["Fecha-I"], "%Y-%m-%d %H:%M:%S")
    min_diff = ((fecha_o - fecha_i).total_seconds()) / 60
    return min_diff


def get_delay_target(data: pd.DataFrame) -> pd.Series:
    data["min_diff"] = data.apply(get_min_diff, axis=1)
    threshold_in_minutes = 15
    data["delay"] = np.where(data["min_diff"] > threshold_in_minutes, 1, 0)

    return data["delay"]


def get_features(data: pd.DataFrame) -> pd.DataFrame:
    features = pd.concat(
        [
            pd.get_dummies(data["OPERA"], prefix="OPERA"),
            pd.get_dummies(data["TIPOVUELO"], prefix="TIPOVUELO"),
            pd.get_dummies(data["MES"], prefix="MES"),
        ],
        axis=1,
    )

    features = features[FEATURES_COLS]
    return features


class DelayModel:
    def __init__(self):
        self._model = None  # Model should be saved in this attribute.

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
        # process default target column
        default_target = "delay"

        if target_column:
            self.target = target_column
        else:
            self.target = default_target

        # get specific target computation function
        if self.target == "delay":
            y = get_delay_target(data)  # needs conversion to dataframe
        else:
            raise NotImplementedError("Only implemented 'delay' as target column")

        y = y.to_frame()

        x = get_features(data)

        return (x, y) if target_column else x

    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        n_y0 = len(target[target[self.target] == 0])
        n_y1 = len(target[target[self.target] == 1])

        self._model = LogisticRegression(
            class_weight={1: n_y0 / len(target), 0: n_y1 / len(target)}
        )
        self._model.fit(features, target[self.target])

        return self

    def predict(self, features: pd.DataFrame) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.

        Returns:
            (List[int]): predicted targets.
        """
        if self._model is None:
            raise RuntimeError("Model not fitted before calling .predict")

        return self._model.predict(features).tolist()
