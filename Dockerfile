# syntax=docker/dockerfile:1.2
FROM python:3.10.11
# put you docker configuration here

WORKDIR /module
RUN apt update && apt install -y build-essential
RUN pip install poetry

# copy all stuff into container
COPY challenge/ ./challenge
COPY pyproject.toml/ .
COPY poetry.lock .
COPY models/ ./models

# add empty README and install
RUN touch README.md
RUN poetry config virtualenvs.create false
RUN poetry install --no-dev


CMD ["python", "-m", "uvicorn", "challenge.api:app", "--host", "0.0.0.0", "--port", "8080"]