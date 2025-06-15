FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN apt update && apt install -y build-essential libopenblas-dev

RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python3", "kdd_qml.py"]