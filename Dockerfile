FROM python:3.12.9-slim

COPY requirements.txt requirements.txt
COPY predicting_loan predicting_loan
COPY models models
COPY setup.py setup.py

RUN pip install --upgrade pip
RUN pip install -e .

# Run container locally
#CMD uvicorn predicting-loan.api_file:app --reload --host 0.0.0.0

# Run container deployed -> GCP
CMD uvicorn predicting_loan.api_file:app --reload --host 0.0.0.0 --port $PORT
