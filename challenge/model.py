import os
import pickle
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

    return data["delay"].to_frame()


def get_features(data: pd.DataFrame) -> pd.DataFrame:
    # get the one hot enconding of the columns suggested by the DS
    features = pd.concat(
        [
            pd.get_dummies(data["OPERA"], prefix="OPERA"),
            pd.get_dummies(data["TIPOVUELO"], prefix="TIPOVUELO"),
            pd.get_dummies(data["MES"], prefix="MES"),
        ],
        axis=1,
    )
    present_features = list(set(FEATURES_COLS).intersection(set(features.columns)))
    missing_features = list(set(FEATURES_COLS).difference(set(features.columns)))

    features = features[present_features]

    # fill missing features with 0 due to one-hot encoding of features
    for feature in missing_features:
        features[feature] = 0

    return features


class DelayModel:
    def __init__(self):
        self._model = LogisticRegression()

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
        x = get_features(data)

        # return different sets, depending on the target
        if target_column is None:
            return x
        elif target_column == "delay":
            y = get_delay_target(data)
            return (x, y)
        else:
            raise NotImplementedError("Only implemented 'delay' as target column")

    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """
        Fit model with preprocessed data.

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
        Predict delays for new flights.

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


if __name__ == "__main__":
    from sklearn.metrics import classification_report, mean_squared_error
    from sklearn.model_selection import train_test_split

    # perform a training of the model for production deployment
    model = DelayModel()
    data = pd.read_csv(filepath_or_buffer="data/data.csv")

    features, target = model.preprocess(data=data, target_column="delay")

    _, features_validation, _, target_validation = train_test_split(
        features, target, test_size=0.33, random_state=42
    )

    model.fit(features=features, target=target)

    predicted_target = model.predict(features_validation)

    report = classification_report(
        target_validation, predicted_target, output_dict=True
    )

    import ipdb

    ipdb.set_trace()

    # save
    model.save_model("challenge/models")

    # re instantiate to override model
    model = DelayModel()
    model.load_model("challenge/models")

    predicted_target_load = model.predict(features_validation)

    print(
        "The difference in prediction is:"
        f" {mean_squared_error(predicted_target, predicted_target_load)}"
    )
