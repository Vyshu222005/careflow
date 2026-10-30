# CareFlow 2.0 dashboard
#
#   docker build -t careflow .
#   docker run -p 8501:8501 careflow
#
# On the first start the container downloads the public dataset and builds the processed data
# (about a minute). Mount a volume on /app/data to keep it between runs:
#   docker run -p 8501:8501 -v careflow-data:/app/data careflow
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 STREAMLIT_SERVER_HEADLESS=true STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

RUN apt-get update && apt-get install -y --no-install-recommends graphviz curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN chmod +x docker/entrypoint.sh

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s CMD curl -fs http://localhost:8501/_stcore/health || exit 1
ENTRYPOINT ["docker/entrypoint.sh"]
