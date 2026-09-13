FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY lab ./lab
COPY static ./static
ENV AUK_LAB_DATA=/data PYTHONUNBUFFERED=1
EXPOSE 7865
CMD ["uvicorn", "lab.app:app", "--host", "0.0.0.0", "--port", "7865", "--workers", "1"]
