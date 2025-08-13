
# Simple Streamlit service for DigitalOcean App Platform
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1     PYTHONUNBUFFERED=1     PIP_NO_CACHE_DIR=1

# System deps (ca-certs, curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends     ca-certificates curl &&     rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# App Platform sets $PORT; Streamlit binds to it
EXPOSE 8080
CMD ["bash", "-lc", "streamlit run app_streamlit.py --server.port $PORT --server.address 0.0.0.0"]
