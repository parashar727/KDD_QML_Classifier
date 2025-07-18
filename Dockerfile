FROM python:3.10-slim

WORKDIR /app

RUN apt update && apt install -y --no-install-recommends \
    build-essential \
    libopenblas-dev \
    && apt clean \
    && rm -rf /var/lib/apt/lists/

COPY requirements.txt .

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python3", "kdd_qml.py"]