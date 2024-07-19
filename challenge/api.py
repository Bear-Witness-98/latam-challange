import sys

import fastapi
import pandas as pd
from fastapi import HTTPException
from pydantic import BaseModel

from challenge.model import DelayModel


def print_to_file(whatever: any):
    with open("file.txt", "a") as sys.stdout:
        print(whatever)


valid_opera_values = [
    "american airlines",
    "air canada",
    "air france",
    "aeromexico",
    "aerolineas argentinas",
    "austral",
    "avianca",
    "alitalia",
    "british airways",
    "copa air",
    "delta air",
    "gol trans",
    "iberia",
    "k.l.m.",
    "qantas airways",
    "united airlines",
    "grupo latam",
    "sky airline",
    "latin american wings",
    "plus ultra lineas aereas",
    "jetsmart spa",
    "oceanair linhas aereas",
    "lacsa",
]

valid_tipo_vuelo_values = [
    "I",
    "N",
]

valid_mes_values = range(1, 13)


def valid_tipo_vuelo(tipo_vuelo: str) -> bool:
    return tipo_vuelo in valid_tipo_vuelo_values


def valid_opera(opera: str) -> bool:
    return opera in valid_opera_values


def valid_mes(mes_value: int) -> bool:
    return mes_value in valid_mes_values


class Flight(BaseModel):
    OPERA: str
    TIPOVUELO: str
    MES: int


class FlightData(BaseModel):
    flights: list[Flight]


app = fastapi.FastAPI()
model = DelayModel()
model.load_model("models")


def flight_data_to_pandas(flight_data: FlightData) -> pd.DataFrame:
    flight_data_dict = {"OPERA": [], "TIPOVUELO": [], "MES": []}
    for elem in flight_data.flights:
        if not valid_opera(elem.OPERA.lower()):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Value for tipo vuelo not valid. Recieved {elem.OPERA},"
                    f" expected one from {[v for v in valid_opera_values]}"
                ),
            )
        if not valid_tipo_vuelo(elem.TIPOVUELO.capitalize()):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Value for tipo vuelo not valid. Recieved {elem.TIPOVUELO},"
                    f" expected one from {[v for v in valid_tipo_vuelo_values]}"
                ),
            )
        if not valid_mes(elem.MES):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Value for tipo vuelo not valid. Recieved {elem.MES},"
                    f" expected one from {valid_mes_values}"
                ),
            )
        flight_data_dict["OPERA"].append(elem.OPERA)
        flight_data_dict["TIPOVUELO"].append(elem.TIPOVUELO)
        flight_data_dict["MES"].append(elem.MES)

    return pd.DataFrame(flight_data_dict)


@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {"status": "OK"}


@app.post("/predict", status_code=200)
async def post_predict(flight_data: FlightData) -> dict:
    # get data and convert to pandas dataframe

    flight_data_df = flight_data_to_pandas(flight_data)
    preprocessed_data = model.preprocess(flight_data_df)

    column_order = model._model.feature_names_in_
    preprocessed_data = preprocessed_data[column_order]

    pred = model.predict(preprocessed_data)

    return {"predict": pred}
